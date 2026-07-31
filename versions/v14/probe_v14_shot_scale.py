#!/usr/bin/env python3
"""Audit whether V14 B0 leaves a deployable shared shot-scale residual."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import (  # noqa: E402
    b0_human_candidates,
)


DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/shot_scale_audit"
SEQUENCE_INPUTS = {
    "three": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching/v14_b0_identity_matching.json",
        "cache": REPO_ROOT
        / "output/v20_phase1_gt_id_multihuman_consensus/case_cache",
    },
    "dance": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/dance_phase2/case_cache",
    },
    "box": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/box_phase3/case_cache",
    },
}
METHODS = (
    "b0_unit_scale",
    "body_state_scale",
    "layout_state_scale",
    "scene_chamfer_scale",
    "explicit_median_scale",
    "gt_metric_relative_scale",
    "oracle_root_scale",
)
HUMAN_METRICS = (
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
    "scene_trimmed_chamfer_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=tuple(SEQUENCE_INPUTS),
        default=tuple(SEQUENCE_INPUTS),
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--scale_min", type=float, default=0.50)
    parser.add_argument("--scale_max", type=float, default=1.50)
    return parser.parse_args()


def finite_distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            name: float("nan") for name in ("mean", "median", "p90", "p95", "std")
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(np.std(array)),
    }


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return geometry.transform_points(transform, np.asarray(points, dtype=np.float64))


def camera_center(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64)[:3, 3]


def mesh_centroid(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    # The median is less sensitive to hands and occasional extreme vertices.
    return np.median(vertices, axis=0)


def rms_body_radius(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    centered = vertices - mesh_centroid(vertices)[None]
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def positive_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-8:
        return float("nan")
    return float(numerator / denominator)


def clip_scale(value: float, args: argparse.Namespace) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, float(args.scale_min), float(args.scale_max)))


def gt_vertices(
    args: argparse.Namespace, identity: str, frame: int
) -> np.ndarray:
    return geometry.load_obj_vertices(geometry.mesh_path(args, identity, frame)).astype(
        np.float64
    )


def frame_metric_scales(
    args: argparse.Namespace,
    humans: dict[str, dict],
    predicted_pose: np.ndarray,
    camera: int,
    frame: int,
) -> dict:
    pred_w2c = np.linalg.inv(np.asarray(predicted_pose, dtype=np.float64))
    true_w2c = geometry.gt_w2c(args, int(camera), int(frame))
    body, radial, depth = [], [], []
    pred_centers, true_centers = {}, {}
    for identity, human in humans.items():
        predicted_camera = transform_points(pred_w2c, human["vertices"])
        true_camera = transform_points(true_w2c, gt_vertices(args, identity, frame))
        predicted_center = mesh_centroid(predicted_camera)
        true_center = mesh_centroid(true_camera)
        pred_centers[identity] = predicted_center
        true_centers[identity] = true_center
        body.append(
            positive_ratio(rms_body_radius(true_camera), rms_body_radius(predicted_camera))
        )
        radial.append(
            positive_ratio(np.linalg.norm(true_center), np.linalg.norm(predicted_center))
        )
        depth.append(positive_ratio(true_center[2], predicted_center[2]))

    layout_numerator = 0.0
    layout_denominator = 0.0
    for first, second in combinations(sorted(set(pred_centers) & set(true_centers)), 2):
        predicted_distance = float(
            np.linalg.norm(pred_centers[first] - pred_centers[second])
        )
        true_distance = float(np.linalg.norm(true_centers[first] - true_centers[second]))
        layout_numerator += predicted_distance * true_distance
        layout_denominator += predicted_distance * predicted_distance
    layout = (
        layout_numerator / layout_denominator
        if layout_denominator > 1e-8
        else float("nan")
    )
    return {
        "body": float(np.nanmedian(body)) if body else float("nan"),
        "radial": float(np.nanmedian(radial)) if radial else float("nan"),
        "depth": float(np.nanmedian(depth)) if depth else float("nan"),
        "layout": float(layout),
        "per_human_body": body,
        "per_human_radial": radial,
        "per_human_depth": depth,
    }


def body_state_scale(cache: dict) -> float:
    pre = cache["humans"][-2]
    post = cache["humans"][-1]
    values = []
    for identity in sorted(set(pre) & set(post)):
        values.append(
            positive_ratio(
                rms_body_radius(pre[identity]["vertices"]),
                rms_body_radius(post[identity]["vertices"]),
            )
        )
    return float(np.nanmedian(values)) if values else float("nan")


def layout_state_scale(cache: dict, b0: np.ndarray) -> float:
    candidates = b0_human_candidates(cache, b0)
    identities = sorted(candidates)
    numerator = 0.0
    denominator = 0.0
    for first, second in combinations(identities, 2):
        target_distance = float(
            np.linalg.norm(
                candidates[first]["anchor"] - candidates[second]["anchor"]
            )
        )
        post_distance = float(
            np.linalg.norm(
                candidates[first]["post_root"] - candidates[second]["post_root"]
            )
        )
        numerator += post_distance * target_distance
        denominator += post_distance * post_distance
    return numerator / denominator if denominator > 1e-8 else float("nan")


def scaled_aligned_points(
    points: np.ndarray,
    post_pose: np.ndarray,
    b0: np.ndarray,
    scale: float,
) -> np.ndarray:
    aligned = transform_points(b0, points)
    aligned_camera = transform_points(
        b0, camera_center(post_pose).reshape(1, 3)
    )[0]
    return aligned_camera[None] + float(scale) * (aligned - aligned_camera[None])


def trimmed_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 16 or len(second) < 16:
        return float("nan")
    forward = cKDTree(first).query(second, k=1, workers=-1)[0]
    backward = cKDTree(second).query(first, k=1, workers=-1)[0]
    values = np.r_[forward, backward]
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    values = values[values <= np.percentile(values, 80)]
    return float(np.mean(values))


def scene_state_scale(cache: dict, b0: np.ndarray, args: argparse.Namespace) -> float:
    target_parts = [np.asarray(value) for value in cache["clouds"][:-1] if len(value)]
    source = np.asarray(cache["clouds"][-1], dtype=np.float64)
    if not target_parts or len(source) < 16:
        return float("nan")
    target = np.concatenate(target_parts, axis=0)

    def objective(scale: float) -> float:
        mapped = scaled_aligned_points(source, cache["poses"][-1], b0, scale)
        value = trimmed_chamfer(target, mapped)
        return value if np.isfinite(value) else 1e6

    result = minimize_scalar(
        objective,
        bounds=(float(args.scale_min), float(args.scale_max)),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 40},
    )
    return float(result.x) if result.success else float("nan")


def target_geometry(args: argparse.Namespace, cache: dict) -> dict[str, dict]:
    case = cache["case"]
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gt_pre_pose = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    old_from_gt = pre_pose @ np.linalg.inv(gt_pre_pose)
    post_frame = int(case["post_frame"])
    output = {}
    for identity in cache["humans"][-1]:
        true_world = gt_vertices(args, identity, post_frame)
        target_vertices = transform_points(old_from_gt, true_world)
        gt_payload = cache["gt"]["post_humans"][identity]
        output[identity] = {
            "vertices": target_vertices,
            "joints": transform_points(old_from_gt, gt_payload["joints"]),
            "root": transform_points(old_from_gt, gt_payload["root"][None])[0],
        }
    return output


def oracle_root_scale(
    cache: dict, b0: np.ndarray, targets: dict[str, dict]
) -> float:
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    aligned_camera = transform_points(
        b0, camera_center(post_pose).reshape(1, 3)
    )[0]
    numerator = 0.0
    denominator = 0.0
    for identity in sorted(set(cache["humans"][-1]) & set(targets)):
        source = transform_points(
            b0, cache["humans"][-1][identity]["root"][None]
        )[0] - aligned_camera
        target = targets[identity]["root"] - aligned_camera
        numerator += float(np.dot(source, target))
        denominator += float(np.dot(source, source))
    return numerator / denominator if denominator > 1e-8 else float("nan")


def evaluate_scale(
    cache: dict,
    b0: np.ndarray,
    targets: dict[str, dict],
    scale: float,
) -> dict:
    scale = float(scale)
    roots, joints, vertices = [], [], []
    predicted_roots, target_roots = {}, {}
    predicted_vertices, target_vertices = {}, {}
    for identity in sorted(set(cache["humans"][-1]) & set(targets)):
        human = cache["humans"][-1][identity]
        mapped_root = scaled_aligned_points(
            human["root"][None], cache["poses"][-1], b0, scale
        )[0]
        mapped_joints = scaled_aligned_points(
            human["joints"], cache["poses"][-1], b0, scale
        )
        mapped_vertices = scaled_aligned_points(
            human["vertices"], cache["poses"][-1], b0, scale
        )
        target = targets[identity]
        roots.append(float(np.linalg.norm(mapped_root - target["root"])))
        count = min(len(mapped_joints), len(target["joints"]))
        joints.append(
            float(
                np.mean(
                    np.linalg.norm(
                        mapped_joints[:count] - target["joints"][:count], axis=1
                    )
                )
            )
        )
        count = min(len(mapped_vertices), len(target["vertices"]))
        vertices.append(
            float(
                np.mean(
                    np.linalg.norm(
                        mapped_vertices[:count] - target["vertices"][:count], axis=1
                    )
                )
            )
        )
        predicted_roots[identity] = mapped_root
        target_roots[identity] = target["root"]
        predicted_vertices[identity] = mapped_vertices
        target_vertices[identity] = target["vertices"]

    pair_distance, pair_vector = [], []
    for first, second in combinations(sorted(predicted_roots), 2):
        pred_vector = predicted_roots[first] - predicted_roots[second]
        true_vector = target_roots[first] - target_roots[second]
        pair_distance.append(abs(np.linalg.norm(pred_vector) - np.linalg.norm(true_vector)))
        pair_vector.append(float(np.linalg.norm(pred_vector - true_vector)))

    target_clouds = [np.asarray(value) for value in cache["clouds"][:-1] if len(value)]
    source_cloud = np.asarray(cache["clouds"][-1], dtype=np.float64)
    scene = float("nan")
    if target_clouds and len(source_cloud):
        scene = trimmed_chamfer(
            np.concatenate(target_clouds, axis=0),
            scaled_aligned_points(source_cloud, cache["poses"][-1], b0, scale),
        )
    return {
        "scale": scale,
        "human_root_error_m": float(np.mean(roots)) if roots else float("nan"),
        "human_joint_error_m": float(np.mean(joints)) if joints else float("nan"),
        "human_vertex_error_m": float(np.mean(vertices)) if vertices else float("nan"),
        "pairwise_distance_error_m": (
            float(np.mean(pair_distance)) if pair_distance else float("nan")
        ),
        "pairwise_vector_error_m": (
            float(np.mean(pair_vector)) if pair_vector else float("nan")
        ),
        "scene_trimmed_chamfer_m": scene,
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def summarize(cases: list[dict], method: str) -> dict:
    rows = [case["methods"][method] for case in cases]
    return {
        "case_count": len(rows),
        "scale": finite_distribution([row["scale"] for row in rows]),
        **{
            metric: finite_distribution([row[metric] for row in rows])
            for metric in HUMAN_METRICS
        },
        "root_improvement_rate_vs_b0": float(
            np.mean(
                [
                    row["human_root_error_m"]
                    < case["methods"]["b0_unit_scale"]["human_root_error_m"]
                    for case, row in zip(cases, rows)
                ]
            )
        ),
    }


def summarize_measurements(cases: list[dict]) -> dict:
    names = (
        "body_relative",
        "radial_relative",
        "depth_relative",
        "layout_metric_relative",
        "body_state",
        "layout_state",
        "scene_state",
        "oracle_root",
    )
    output = {
        name: finite_distribution([case["scale_measurements"][name] for case in cases])
        for name in names
    }
    scale_rows = np.asarray(
        [
            [
                case["scale_measurements"]["body_relative"],
                case["scale_measurements"]["radial_relative"],
                case["scale_measurements"]["layout_metric_relative"],
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    output["gt_cue_pairwise_abs_disagreement"] = {
        "body_vs_radial": finite_distribution(
            np.abs(scale_rows[:, 0] - scale_rows[:, 1]).tolist()
        ),
        "body_vs_layout": finite_distribution(
            np.abs(scale_rows[:, 0] - scale_rows[:, 2]).tolist()
        ),
        "radial_vs_layout": finite_distribution(
            np.abs(scale_rows[:, 1] - scale_rows[:, 2]).tolist()
        ),
    }
    return output


def process_case(
    args: argparse.Namespace,
    sequence: str,
    cache: dict,
    b0: np.ndarray,
) -> dict:
    case = cache["case"]
    pre_frame = int(case["pre_frames"][-1])
    post_frame = int(case["post_frame"])
    pre_metric = frame_metric_scales(
        args,
        cache["humans"][-2],
        cache["poses"][-2],
        int(case["source_camera"]),
        pre_frame,
    )
    post_metric = frame_metric_scales(
        args,
        cache["humans"][-1],
        cache["poses"][-1],
        int(case["target_camera"]),
        post_frame,
    )
    relative = {
        name: positive_ratio(post_metric[name], pre_metric[name])
        for name in ("body", "radial", "depth", "layout")
    }
    body_state = body_state_scale(cache)
    layout_state = layout_state_scale(cache, b0)
    scene_state = scene_state_scale(cache, b0, args)
    targets = target_geometry(args, cache)
    oracle_root = oracle_root_scale(cache, b0, targets)
    explicit_values = [body_state, layout_state, scene_state]
    explicit_values = [value for value in explicit_values if np.isfinite(value)]
    explicit_median = float(np.median(explicit_values)) if explicit_values else 1.0
    gt_metric_relative = float(np.nanmedian([relative["radial"], relative["depth"]]))
    scales = {
        "b0_unit_scale": 1.0,
        "body_state_scale": clip_scale(body_state, args),
        "layout_state_scale": clip_scale(layout_state, args),
        "scene_chamfer_scale": clip_scale(scene_state, args),
        "explicit_median_scale": clip_scale(explicit_median, args),
        "gt_metric_relative_scale": clip_scale(gt_metric_relative, args),
        "oracle_root_scale": clip_scale(oracle_root, args),
    }
    return {
        "sequence": sequence,
        "case": case,
        "scale_measurements": {
            "pre_metric": pre_metric,
            "post_metric": post_metric,
            "body_relative": relative["body"],
            "radial_relative": relative["radial"],
            "depth_relative": relative["depth"],
            "layout_metric_relative": relative["layout"],
            "body_state": body_state,
            "layout_state": layout_state,
            "scene_state": scene_state,
            "oracle_root": oracle_root,
        },
        "methods": {
            name: evaluate_scale(cache, b0, targets, scale)
            for name, scale in scales.items()
        },
    }


def load_sequence(args: argparse.Namespace, sequence: str) -> list[dict]:
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    args.sequence = sequence
    inputs = SEQUENCE_INPUTS[sequence]
    report = json.loads(inputs["report"].read_text(encoding="utf-8"))
    report_cases = report["cases"][: int(args.max_cases) or None]
    output = []
    for index, report_case in enumerate(report_cases, start=1):
        case = report_case["case"]
        cache = torch.load(
            inputs["cache"] / f"{case['key']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        cache = geometry.reassign_cache_gt_identities(
            SimpleNamespace(
                data_root=args.data_root,
                size=int(args.size),
                sequence=sequence,
            ),
            cache,
        )
        b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
        output.append(process_case(args, sequence, cache, b0))
        print(
            f"[{sequence} {index:03d}/{len(report_cases):03d}] {case['key']}",
            flush=True,
        )
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Shot-Scale Audit",
        "",
        "All scales are one causal post-shot scalar applied around the first post-cut "
        "camera center to camera translation, pointmap, and every human.",
        "",
        "## Overall",
        "",
        "| Method | Scale median | Root | Joint | Vertex | Layout distance | Scene | Root improve |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["summary"]["all"]["methods"][method]
        lines.append(
            f"| {method} | {row['scale']['median']:.3f} | "
            f"{row['human_root_error_m']['mean']:.3f} | "
            f"{row['human_joint_error_m']['mean']:.3f} | "
            f"{row['human_vertex_error_m']['mean']:.3f} | "
            f"{row['pairwise_distance_error_m']['mean']:.3f} | "
            f"{row['scene_trimmed_chamfer_m']['mean']:.3f} | "
            f"{100.0 * row['root_improvement_rate_vs_b0']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Scale Measurements",
            "",
            "| Cue | Mean | Median | P90 | Std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    measurements = report["summary"]["all"]["measurements"]
    for name in (
        "body_relative",
        "radial_relative",
        "depth_relative",
        "layout_metric_relative",
        "body_state",
        "layout_state",
        "scene_state",
        "oracle_root",
    ):
        row = measurements[name]
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['median']:.3f} | "
            f"{row['p90']:.3f} | {row['std']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for sequence in args.sequences:
        cases.extend(load_sequence(args, sequence))
    groups = {"all": cases}
    groups.update(
        {
            sequence: [case for case in cases if case["sequence"] == sequence]
            for sequence in args.sequences
        }
    )
    report = {
        "experiment": "V14 shared shot-scale audit",
        "protocol": {
            "sequences": list(args.sequences),
            "case_count": len(cases),
            "scale_bounds": [float(args.scale_min), float(args.scale_max)],
            "identity": "strict GT mesh-projection reassignment for WHERE isolation",
            "scale_application": (
                "one similarity about first post camera center; camera trajectory, "
                "pointmap, roots, joints, vertices share the same scalar"
            ),
            "camera_note": (
                "first post camera center/rotation are unchanged by scale; later "
                "within-shot camera translations would be scaled"
            ),
        },
        "summary": {
            name: {
                "case_count": len(values),
                "measurements": summarize_measurements(values),
                "methods": {method: summarize(values, method) for method in METHODS},
            }
            for name, values in groups.items()
        },
        "cases": cases,
    }
    json_path = args.output_dir / "v14_shot_scale_audit.json"
    md_path = args.output_dir / "v14_shot_scale_audit.md"
    json_path.write_text(
        json.dumps(serializable(report), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
