#!/usr/bin/env python3
"""Minimal person-conditioned DA3 depth probe with bit-exact frozen B0 camera.

The candidate sees only predicted Human3R people/boxes, a frozen B0, DA3 pair
depth/confidence, and the existing GT-free geometry matcher used as identity
memory.  GT camera/body fields are accessed only after every person correction
has been frozen.  The current cache has no predicted per-person raster mask, so
DA3 depth is sampled from a conservative torso core of each predicted bbox.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset/Depth-Anything-3"
for path in (REPO_ROOT, REPO_ROOT / "src", DA3_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_sift_epipolar import FrameReader  # noqa: E402


DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_MODEL = DA3_ROOT / "checkpoints/DAE-base"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_da3_person_pointmap"
)
METHODS = (
    "b0",
    "pointmap_memory_translation_cap030",
    "pointmap_memory_similarity_cap030",
    "oracle_gt_ray_translation",
    "oracle_gt_ray_similarity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--depth_quantile", type=float, default=0.35)
    parser.add_argument("--cap_m", type=float, default=0.30)
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--max_relative_mad", type=float, default=0.20)
    parser.add_argument("--max_core_overlap", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def homogeneous(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (4, 4):
        return value
    output = np.eye(4, dtype=np.float64)
    output[:3] = value
    return output


def bbox_core_mask(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Conservative torso/pelvis core; bbox coordinates are in 512-square input."""
    bbox = np.asarray(bbox, dtype=np.float64) * np.asarray(
        [width / 512.0, height / 512.0, width / 512.0, height / 512.0]
    )
    center_x = 0.5 * (bbox[0] + bbox[2])
    box_width = max(float(bbox[2] - bbox[0]), 1.0)
    box_height = max(float(bbox[3] - bbox[1]), 1.0)
    lower = np.floor(
        [center_x - 0.20 * box_width, bbox[1] + 0.25 * box_height]
    ).astype(int)
    upper = np.ceil(
        [center_x + 0.20 * box_width, bbox[1] + 0.72 * box_height]
    ).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, [width, height])
    mask = np.zeros((height, width), dtype=bool)
    if upper[0] > lower[0] and upper[1] > lower[1]:
        mask[lower[1] : upper[1], lower[0] : upper[0]] = True
    return mask


def range_map(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = (xx - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) / intrinsic[1, 1]
    return np.asarray(depth, dtype=np.float64) * np.sqrt(x * x + y * y + 1.0)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    threshold = float(quantile) * float(cumulative[-1])
    return float(values[min(int(np.searchsorted(cumulative, threshold)), len(values) - 1)])


def region_observation(
    ranges: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    quantile: float,
    min_pixels: int,
) -> dict:
    valid = mask & np.isfinite(ranges) & (ranges > 0.02) & np.isfinite(confidence)
    if int(valid.sum()) < min_pixels:
        return {"valid": False, "reason": "too_few_core_pixels", "pixel_count": int(valid.sum())}
    conf_threshold = float(np.percentile(confidence[valid], 30.0))
    valid &= confidence >= conf_threshold
    values = ranges[valid]
    weights = np.maximum(confidence[valid].astype(np.float64), 1e-6)
    estimate = weighted_quantile(values, weights, quantile)
    mad = float(np.median(np.abs(values - np.median(values))))
    return {
        "valid": True,
        "range_units": estimate,
        "pixel_count": int(len(values)),
        "confidence_mean": float(weights.mean()),
        "relative_mad": mad / max(abs(estimate), 1e-8),
    }


def apply_ray_change(
    root: np.ndarray,
    joints: np.ndarray,
    vertices: np.ndarray,
    camera_center: np.ndarray,
    ray: np.ndarray,
    delta: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if kind == "translation":
        shift = delta * ray
        return root + shift, joints + shift, vertices + shift, 1.0
    current_range = float(np.linalg.norm(root - camera_center))
    scale = (current_range + delta) / max(current_range, 1e-8)
    scale = max(scale, 0.05)
    transform = lambda points: camera_center + scale * (points - camera_center)
    return transform(root), transform(joints), transform(vertices), float(scale)


def point_errors(predicted: tuple[np.ndarray, np.ndarray, np.ndarray], target: dict) -> dict:
    root, joints, vertices = predicted[:3]
    target_root = np.asarray(target["root"], dtype=np.float64)
    target_joints = np.asarray(target["joints"], dtype=np.float64)
    target_vertices = np.asarray(target["vertices"], dtype=np.float64)
    joint_count = min(len(joints), len(target_joints))
    vertex_count = min(len(vertices), len(target_vertices))
    return {
        "root_error_m": float(np.linalg.norm(root - target_root)),
        "joint_error_m": float(
            np.linalg.norm(joints[:joint_count] - target_joints[:joint_count], axis=1).mean()
        ),
        "vertex_error_m": float(
            np.linalg.norm(vertices[:vertex_count] - target_vertices[:vertex_count], axis=1).mean()
        ),
    }


def run_da3(
    model: DepthAnything3,
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
    process_res: int,
) -> tuple[dict, float]:
    started = time.perf_counter()
    prediction = model.inference(
        [first_rgb, second_rgb],
        process_res=int(process_res),
        use_ray_pose=False,
        ref_view_strategy="first",
    )
    elapsed = time.perf_counter() - started
    if prediction.depth is None or prediction.intrinsics is None or prediction.extrinsics is None:
        raise RuntimeError("DA3 prediction lacks depth/intrinsics/extrinsics")
    confidence = (
        np.ones_like(prediction.depth, dtype=np.float32)
        if prediction.conf is None
        else np.asarray(prediction.conf, dtype=np.float32)
    )
    return {
        "depth": np.asarray(prediction.depth, dtype=np.float32),
        "confidence": confidence,
        "intrinsics": np.asarray(prediction.intrinsics, dtype=np.float64),
        "extrinsics": np.stack([homogeneous(row) for row in prediction.extrinsics]),
    }, elapsed


def auto_identity_pairs(report_case: dict, cache: dict) -> list[tuple[str, str]]:
    """Use the frozen predicted root+torso matcher, never its GT correctness fields."""
    matcher = report_case["matching"]["learned_b0"]["matchers"]["root_torso"]
    predicted = matcher["predicted_identity_by_pre_identity"]
    return [
        (str(pre), str(post))
        for pre, post in predicted.items()
        if pre in cache["humans"][-2] and post in cache["humans"][-1]
    ]


def build_candidates(
    report_case: dict,
    cache: dict,
    da3: dict,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    """GT-free candidate path. Returned arrays are frozen before evaluation."""
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    final_camera = b0 @ post_pose
    camera_snapshot = final_camera.copy()

    da3_c2w = np.linalg.inv(da3["extrinsics"])
    da3_baseline = float(np.linalg.norm(da3_c2w[1, :3, 3] - da3_c2w[0, :3, 3]))
    frozen_baseline = float(np.linalg.norm(final_camera[:3, 3] - pre_pose[:3, 3]))
    depth_scale = frozen_baseline / max(da3_baseline, 1e-8)
    maps = [
        range_map(da3["depth"][index], da3["intrinsics"][index])
        for index in (0, 1)
    ]
    pairs = auto_identity_pairs(report_case, cache)
    masks = {
        (view_index, identity): bbox_core_mask(
            cache["humans"][-2 + view_index][identity]["bbox"],
            maps[view_index].shape[0],
            maps[view_index].shape[1],
        )
        for view_index in (0, 1)
        for identity in cache["humans"][-2 + view_index]
    }
    proposals, diagnostics = {}, {}
    for pre_identity, post_identity in pairs:
        pre_human = cache["humans"][-2][pre_identity]
        post_human = cache["humans"][-1][post_identity]
        pre_obs = region_observation(
            maps[0], da3["confidence"][0], masks[(0, pre_identity)],
            args.depth_quantile, args.min_pixels,
        )
        post_obs = region_observation(
            maps[1], da3["confidence"][1], masks[(1, post_identity)],
            args.depth_quantile, args.min_pixels,
        )
        other_post = np.zeros_like(masks[(1, post_identity)])
        for identity in cache["humans"][-1]:
            if identity != post_identity:
                other_post |= masks[(1, identity)]
        core = masks[(1, post_identity)]
        overlap = float((core & other_post).sum() / max(int(core.sum()), 1))

        pre_root = np.asarray(pre_human["root"], dtype=np.float64)
        raw_post_root = np.asarray(post_human["root"], dtype=np.float64)
        root = transform_points(b0, raw_post_root[None])[0]
        joints = transform_points(b0, np.asarray(post_human["joints"], dtype=np.float64))
        vertices = transform_points(b0, np.asarray(post_human["vertices"], dtype=np.float64))
        camera_center = final_camera[:3, 3]
        ray_vector = root - camera_center
        current_range = float(np.linalg.norm(ray_vector))
        ray = ray_vector / max(current_range, 1e-8)
        pre_range = float(np.linalg.norm(pre_root - pre_pose[:3, 3]))
        valid = bool(
            pre_obs.get("valid") and post_obs.get("valid")
            and pre_obs["relative_mad"] <= args.max_relative_mad
            and post_obs["relative_mad"] <= args.max_relative_mad
            and overlap <= args.max_core_overlap
            and 0.05 < depth_scale < 20.0
        )
        raw_delta = (
            pre_range
            + depth_scale * (post_obs["range_units"] - pre_obs["range_units"])
            - current_range
            if valid else 0.0
        )
        delta = float(np.clip(raw_delta, -args.cap_m, args.cap_m)) if valid else 0.0
        proposals[post_identity] = {
            "base": (root, joints, vertices, 1.0),
            "translation": apply_ray_change(
                root, joints, vertices, camera_center, ray, delta, "translation"
            ),
            "similarity": apply_ray_change(
                root, joints, vertices, camera_center, ray, delta, "similarity"
            ),
            "ray": ray,
            "camera_center": camera_center,
        }
        diagnostics[post_identity] = {
            "pre_memory_identity": pre_identity,
            "post_detection_identity": post_identity,
            "pre": pre_obs,
            "post": post_obs,
            "core_overlap_fraction": overlap,
            "accepted": valid,
            "depth_scale_from_frozen_camera_baseline": depth_scale,
            "raw_depth_residual_m": raw_delta,
            "applied_depth_residual_m": delta,
        }
    if not np.array_equal(final_camera, camera_snapshot):
        raise AssertionError("Person proposal mutated frozen camera")
    return {
        "b0": b0,
        "frozen_camera": final_camera,
        "camera_snapshot": camera_snapshot,
        "people": proposals,
    }, diagnostics


def evaluate_frozen(report_case: dict, cache: dict, frozen: dict) -> dict:
    """Evaluation-only path: GT is first accessed in this function."""
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64))
    target_humans = cache["gt"]["post_humans"]
    people = {method: [] for method in METHODS}
    diagnostics = []
    for identity, proposal in frozen["people"].items():
        if identity not in target_humans:
            continue
        target = {
            key: transform_points(gauge, np.asarray(target_humans[identity][key]))
            for key in ("root", "joints", "vertices")
        }
        base = proposal["base"]
        people["b0"].append({"identity": identity, **point_errors(base, target)})
        for kind in ("translation", "similarity"):
            method = f"pointmap_memory_{kind}_cap030"
            people[method].append(
                {"identity": identity, **point_errors(proposal[kind], target)}
            )
        root, joints, vertices = base[:3]
        ray = proposal["ray"]
        oracle_delta = float(np.dot(target["root"] - root, ray))
        error = root - target["root"]
        radial = float(np.dot(error, ray))
        tangential = float(np.linalg.norm(error - radial * ray))
        for kind in ("translation", "similarity"):
            corrected = apply_ray_change(
                root, joints, vertices, proposal["camera_center"], ray,
                oracle_delta, kind,
            )
            method = f"oracle_gt_ray_{kind}"
            people[method].append(
                {"identity": identity, **point_errors(corrected, target)}
            )
        diagnostics.append(
            {
                "identity": identity,
                "b0_root_error_m": float(np.linalg.norm(error)),
                "radial_error_m": radial,
                "abs_radial_error_m": abs(radial),
                "tangential_error_m": tangential,
                "oracle_depth_residual_m": oracle_delta,
            }
        )
    return {"methods": people, "gt_ray_diagnostics": diagnostics}


def finite_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
        "p90": float(np.percentile(array, 90)) if len(array) else float("nan"),
    }


def summarize(cases: list[dict]) -> dict:
    output = {
        "case_count": len(cases),
        "person_count": int(sum(len(case["evaluation"]["methods"]["b0"]) for case in cases)),
        "accepted_person_count": int(
            sum(sum(row["accepted"] for row in case["candidate_diagnostics"].values()) for case in cases)
        ),
        "methods": {},
    }
    baseline = [person for case in cases for person in case["evaluation"]["methods"]["b0"]]
    for method in METHODS:
        rows = [person for case in cases for person in case["evaluation"]["methods"][method]]
        summary = {
            metric: finite_stats([row[metric] for row in rows])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        deltas = [row["root_error_m"] - base["root_error_m"] for row, base in zip(rows, baseline)]
        summary["root_mean_delta_m"] = float(np.mean(deltas)) if deltas else float("nan")
        summary["root_improvement_rate"] = float(np.mean(np.asarray(deltas) < 0)) if deltas else float("nan")
        output["methods"][method] = summary
    output["camera_bit_exact_all"] = bool(
        all(case["camera_bit_exact"] for case in cases)
    )
    residual_pairs = []
    for case in cases:
        oracle = {
            row["identity"]: row["oracle_depth_residual_m"]
            for row in case["evaluation"]["gt_ray_diagnostics"]
        }
        for identity, row in case["candidate_diagnostics"].items():
            if row["accepted"] and identity in oracle:
                residual_pairs.append(
                    [row["raw_depth_residual_m"], oracle[identity]]
                )
    residual_pairs = np.asarray(residual_pairs, dtype=np.float64).reshape(-1, 2)
    output["pointmap_vs_gt_ray"] = {
        "accepted_pair_count": int(len(residual_pairs)),
        "sign_agreement_rate": (
            float(np.mean(np.sign(residual_pairs[:, 0]) == np.sign(residual_pairs[:, 1])))
            if len(residual_pairs) else float("nan")
        ),
        "pearson_correlation": (
            float(np.corrcoef(residual_pairs.T)[0, 1])
            if len(residual_pairs) >= 2 else float("nan")
        ),
        "pairs_predicted_oracle_m": residual_pairs,
    }
    return output


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# B0 + DA3 Person Pointmap — Minimal 3-Cut Dev Probe",
        "",
        "B0 camera is frozen bit-exact. The current cache has no predicted per-person mask; "
        "the candidate uses a conservative torso/pelvis core of the predicted bbox.",
        "",
        f"Cases/people: `{summary['case_count']}/{summary['person_count']}`; accepted person observations: "
        f"`{summary['accepted_person_count']}`; camera bit-exact: `{summary['camera_bit_exact_all']}`.",
        "",
        "| Method | Root | Joint | Vertex | Root delta | Improve |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['root_error_m']['mean']:.4f} | "
            f"{row['joint_error_m']['mean']:.4f} | {row['vertex_error_m']['mean']:.4f} | "
            f"{row['root_mean_delta_m']:+.4f} | {row['root_improvement_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The pointmap candidate transfers each matched person's DA3 pre→post surface-range "
            "change onto the old Human3R identity root range, then clips the ray correction to 0.30 m. "
            "GT-ray methods are evaluation-only upper bounds.",
            "",
            f"For accepted people, pointmap-vs-oracle depth-residual sign agreement is "
            f"`{summary['pointmap_vs_gt_ray']['sign_agreement_rate']:.1%}` and Pearson correlation is "
            f"`{summary['pointmap_vs_gt_ray']['pearson_correlation']:.3f}`. This rejects the current "
            "bbox-core surface-change estimator even though the GT-ray upper bound is strong.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not (args.model_path / "model.safetensors").is_file():
        raise FileNotFoundError(args.model_path / "model.safetensors")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(SEQUENCE_INPUTS["three"]["report"].read_text(encoding="utf-8"))
    selected = source["cases"][: int(args.max_cases)]
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    reader = FrameReader(args, "three")
    cases, failures = [], []
    try:
        for index, report_case in enumerate(selected, start=1):
            key = report_case["case"]["key"]
            path = cases_dir / f"{key}.json"
            try:
                cache = torch.load(
                    SEQUENCE_INPUTS["three"]["cache"] / f"{key}.pt",
                    map_location="cpu", weights_only=False,
                )
                first = reader.read(
                    int(report_case["case"]["source_camera"]),
                    int(report_case["case"]["pre_frames"][-1]),
                )[..., ::-1].copy()
                second = reader.read(
                    int(report_case["case"]["target_camera"]),
                    int(report_case["case"]["post_frame"]),
                )[..., ::-1].copy()
                da3, elapsed = run_da3(model, first, second, args.process_res)
                frozen, candidate_diagnostics = build_candidates(report_case, cache, da3, args)
                evaluation = evaluate_frozen(report_case, cache, frozen)
                case = {
                    "status": "ok",
                    "case": report_case["case"],
                    "mask_source": "predicted_bbox_conservative_torso_core_no_person_mask_available",
                    "camera_bit_exact": bool(
                        np.array_equal(frozen["frozen_camera"], frozen["camera_snapshot"])
                    ),
                    "frozen_b0": frozen["b0"],
                    "frozen_camera": frozen["frozen_camera"],
                    "candidate_diagnostics": candidate_diagnostics,
                    "evaluation": evaluation,
                    "da3_seconds": elapsed,
                }
                cases.append(case)
                print(
                    f"[{index}/{len(selected)}] {key} people={len(frozen['people'])} "
                    f"accepted={sum(row['accepted'] for row in candidate_diagnostics.values())} "
                    f"camera_exact={case['camera_bit_exact']}", flush=True,
                )
            except Exception as error:
                case = {
                    "status": "failed", "case_key": key, "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(case)
            path.write_text(
                json.dumps(jsonable(case), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
                encoding="utf-8",
            )
    finally:
        reader.close()
    report = {
        "experiment": "v14_b0_da3_person_pointmap_minimal_three_dev",
        "protocol": {
            "camera": "frozen learned B0; candidate cannot modify camera",
            "candidate_inputs": "Human3R predicted people/bboxes, DA3 depth/confidence/intrinsics/extrinsics, frozen predicted root+torso identity matcher",
            "gt_candidate_or_gate_usage": False,
            "gt_usage": "evaluate frozen per-person geometry and GT-ray oracle only",
            "mask_limitation": "no predicted per-person mask in cache; conservative predicted bbox torso core",
        },
        "parameters": {
            "max_cases": int(args.max_cases), "process_res": int(args.process_res),
            "depth_quantile": float(args.depth_quantile), "cap_m": float(args.cap_m),
            "min_pixels": int(args.min_pixels), "max_relative_mad": float(args.max_relative_mad),
            "max_core_overlap": float(args.max_core_overlap),
        },
        "summary": summarize(cases),
        "failures": failures,
        "cases": cases,
    }
    json_path = args.output_dir / "v14_b0_da3_person_pointmap.json"
    md_path = args.output_dir / "v14_b0_da3_person_pointmap.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    md_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f">> {json_path}", flush=True)


if __name__ == "__main__":
    main()
