#!/usr/bin/env python3
"""Combine V14 shared-scale candidates with frozen B0-centered human refinement."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

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
    SEQUENCE_INPUTS,
    body_state_scale,
    camera_center,
    clip_scale,
    finite_distribution,
    layout_state_scale,
    scene_state_scale,
    target_geometry,
    transform_points,
)
from versions.v14.run_v14_2_multihuman_sequence import (  # noqa: E402
    b0_human_candidates,
)


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/scale_refinement_audit"
SCALE_MODES = ("unit", "explicit", "oracle")
REFINEMENTS = ("b0", "rotation_only", "translation_only", "full_multi")
METHODS = tuple(
    f"{scale}_{refinement}"
    for scale in SCALE_MODES
    for refinement in REFINEMENTS
)
METRICS = (
    "camera_translation_error_m",
    "camera_rotation_error_deg",
    "camera_composite",
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
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
    parser.add_argument("--scale_report", type=Path, default=REPO_ROOT / "output/v14/shot_scale_audit/v14_shot_scale_audit.json")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--scale_min", type=float, default=0.50)
    parser.add_argument("--scale_max", type=float, default=1.50)
    return parser.parse_args()


def similarity_map(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return float(scale) * (points @ np.asarray(rotation, dtype=np.float64).T) + np.asarray(
        translation, dtype=np.float64
    )[None]


def target_camera(cache: dict) -> np.ndarray:
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    old_from_gt = pre_pose @ np.linalg.inv(np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64))
    return old_from_gt @ np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return geometry.rotation_distance_deg(np.asarray(first), np.asarray(second))


def evaluate_similarity(
    cache: dict,
    targets: dict[str, dict],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict:
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    predicted_camera_center = (
        float(scale) * np.asarray(rotation) @ camera_center(post_pose)
        + np.asarray(translation)
    )
    predicted_camera_rotation = np.asarray(rotation) @ post_pose[:3, :3]
    desired_camera = target_camera(cache)
    camera_t = float(
        np.linalg.norm(predicted_camera_center - camera_center(desired_camera))
    )
    camera_r = rotation_error_deg(
        predicted_camera_rotation, desired_camera[:3, :3]
    )

    root_errors, joint_errors, vertex_errors = [], [], []
    predicted_roots, target_roots = {}, {}
    for identity in sorted(set(cache["humans"][-1]) & set(targets)):
        human = cache["humans"][-1][identity]
        mapped_root = similarity_map(
            human["root"][None], scale, rotation, translation
        )[0]
        mapped_joints = similarity_map(
            human["joints"], scale, rotation, translation
        )
        mapped_vertices = similarity_map(
            human["vertices"], scale, rotation, translation
        )
        target = targets[identity]
        root_errors.append(float(np.linalg.norm(mapped_root - target["root"])))
        joint_count = min(len(mapped_joints), len(target["joints"]))
        vertex_count = min(len(mapped_vertices), len(target["vertices"]))
        joint_errors.append(
            float(
                np.linalg.norm(
                    mapped_joints[:joint_count] - target["joints"][:joint_count],
                    axis=1,
                ).mean()
            )
        )
        vertex_errors.append(
            float(
                np.linalg.norm(
                    mapped_vertices[:vertex_count]
                    - target["vertices"][:vertex_count],
                    axis=1,
                ).mean()
            )
        )
        predicted_roots[identity] = mapped_root
        target_roots[identity] = target["root"]

    pair_distance, pair_vector = [], []
    for first, second in combinations(sorted(predicted_roots), 2):
        predicted = predicted_roots[first] - predicted_roots[second]
        target = target_roots[first] - target_roots[second]
        pair_distance.append(abs(np.linalg.norm(predicted) - np.linalg.norm(target)))
        pair_vector.append(float(np.linalg.norm(predicted - target)))
    return {
        "scale": float(scale),
        "camera_translation_error_m": camera_t,
        "camera_rotation_error_deg": camera_r,
        "camera_composite": camera_t + 0.02 * camera_r,
        "human_root_error_m": float(np.mean(root_errors)),
        "human_joint_error_m": float(np.mean(joint_errors)),
        "human_vertex_error_m": float(np.mean(vertex_errors)),
        "pairwise_distance_error_m": float(np.mean(pair_distance)) if pair_distance else float("nan"),
        "pairwise_vector_error_m": float(np.mean(pair_vector)) if pair_vector else float("nan"),
        "catastrophic": bool(camera_t > 2.0 or camera_r > 45.0),
    }


def solutions_for_scale(cache: dict, b0: np.ndarray, scale: float) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    candidates = b0_human_candidates(cache, b0)
    identities = sorted(candidates)
    rotations = [candidates[identity]["rotation"] for identity in identities]
    rotation_mean = geometry.so3_mean(rotations)
    b0_rotation = np.asarray(b0, dtype=np.float64)[:3, :3]
    post_center = camera_center(cache["poses"][-1])
    b0_center = transform_points(b0, post_center[None])[0]

    def keep_camera_center(rotation: np.ndarray) -> np.ndarray:
        return b0_center - float(scale) * np.asarray(rotation) @ post_center

    translations_at_b0 = [
        candidates[identity]["anchor"]
        - float(scale) * b0_rotation @ candidates[identity]["post_root"]
        for identity in identities
    ]
    translations_full = [
        candidates[identity]["anchor"]
        - float(scale) * candidates[identity]["rotation"] @ candidates[identity]["post_root"]
        for identity in identities
    ]
    return {
        "b0": (scale, b0_rotation, keep_camera_center(b0_rotation)),
        "rotation_only": (
            scale,
            rotation_mean,
            keep_camera_center(rotation_mean),
        ),
        "translation_only": (
            scale,
            b0_rotation,
            np.mean(np.stack(translations_at_b0), axis=0),
        ),
        "full_multi": (
            scale,
            rotation_mean,
            np.mean(np.stack(translations_full), axis=0),
        ),
    }


def summarize(cases: list[dict], method: str) -> dict:
    rows = [case["methods"][method] for case in cases]
    return {
        "case_count": len(rows),
        "scale": finite_distribution([row["scale"] for row in rows]),
        **{
            metric: finite_distribution([row[metric] for row in rows])
            for metric in METRICS
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows])),
        "composite_improvement_rate_vs_unit_b0": float(
            np.mean(
                [
                    row["camera_composite"]
                    < case["methods"]["unit_b0"]["camera_composite"]
                    for case, row in zip(cases, rows)
                ]
            )
        ),
    }


def explicit_scale_for_case(scale_case: dict) -> float:
    values = [
        scale_case["scale_measurements"][name]
        for name in ("body_state", "layout_state", "scene_state")
        if np.isfinite(scale_case["scale_measurements"][name])
    ]
    return float(np.median(values)) if values else 1.0


def process_case(args: argparse.Namespace, sequence: str, cache: dict, b0: np.ndarray, scale_case: dict) -> dict:
    scales = {
        "unit": 1.0,
        "explicit": clip_scale(explicit_scale_for_case(scale_case), args),
        "oracle": clip_scale(scale_case["scale_measurements"]["oracle_root"], args),
    }
    targets = target_geometry(args, cache)
    methods = {}
    for scale_name, scale in scales.items():
        for refinement, (value, rotation, translation) in solutions_for_scale(
            cache, b0, scale
        ).items():
            methods[f"{scale_name}_{refinement}"] = evaluate_similarity(
                cache, targets, value, rotation, translation
            )
    return {
        "sequence": sequence,
        "case": cache["case"],
        "scales": scales,
        "methods": methods,
    }


def load_sequence(args: argparse.Namespace, sequence: str, scale_cases: dict[str, dict]) -> list[dict]:
    args.sequence = sequence
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    inputs = SEQUENCE_INPUTS[sequence]
    report = json.loads(inputs["report"].read_text(encoding="utf-8"))
    rows = report["cases"][: int(args.max_cases) or None]
    output = []
    for index, report_case in enumerate(rows, start=1):
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
        output.append(
            process_case(args, sequence, cache, b0, scale_cases[case["key"]])
        )
        print(f"[{sequence} {index:03d}/{len(rows):03d}] {case['key']}", flush=True)
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Scale + Refinement Audit",
        "",
        "| Method | Scale | Camera T | Camera R | Composite | Root | Vertex | Layout | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["summary"]["all"][method]
        lines.append(
            f"| {method} | {row['scale']['median']:.3f} | "
            f"{row['camera_translation_error_m']['mean']:.3f} | "
            f"{row['camera_rotation_error_deg']['mean']:.2f} | "
            f"{row['camera_composite']['mean']:.3f} | "
            f"{row['human_root_error_m']['mean']:.3f} | "
            f"{row['human_vertex_error_m']['mean']:.3f} | "
            f"{row['pairwise_distance_error_m']['mean']:.3f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    scale_report = json.loads(args.scale_report.read_text(encoding="utf-8"))
    scale_cases = {case["case"]["key"]: case for case in scale_report["cases"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for sequence in args.sequences:
        cases.extend(load_sequence(args, sequence, scale_cases))
    groups = {"all": cases}
    groups.update(
        {
            sequence: [case for case in cases if case["sequence"] == sequence]
            for sequence in args.sequences
        }
    )
    report = {
        "experiment": "V14 shared scale plus B0-centered refinement audit",
        "protocol": {
            "case_count": len(cases),
            "identity": "strict GT identity for WHERE isolation",
            "explicit_scale": "median(body-state, layout-state, scene-Chamfer)",
            "oracle_scale": "GT root optimum around B0 camera center",
            "scale_application": "one shared scalar for camera trajectory, pointmap, and all humans",
        },
        "summary": {
            name: {method: summarize(values, method) for method in METHODS}
            for name, values in groups.items()
        },
        "cases": cases,
    }
    json_path = args.output_dir / "v14_scale_refinement_audit.json"
    md_path = args.output_dir / "v14_scale_refinement_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
