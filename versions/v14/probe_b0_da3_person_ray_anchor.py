#!/usr/bin/env python3
"""GT-ID feasibility probe for per-person ray/depth correction after B0+DA3.

Camera alignment is frozen and never changed. For every persistent person, the
pre-cut Human3R world trajectory predicts an anchor. The first post-cut
Human3R root defines a ray from the already aligned post camera. We test moving
the person only along that ray, either by rigid translation or by a
camera-centered similarity that exactly preserves its image projection.

GT identity and GT geometry are evaluation-only. This is a WHERE/feasibility
probe, not yet a deployable association method.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


THREE_REPORT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/da3_shared_pose_three_dev/"
    "v14_b0_da3_shared_pose.json"
)
FROZEN_REPORT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/da3_shared_pose_dance_box_frozen/"
    "v14_b0_da3_shared_pose.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/b0_da3_person_ray_anchor"
)
CAPS_M = (0.10, 0.20, 0.30, 0.50, 1.00)
GATES_M = (0.10, 0.20, 0.30, 0.50, 1.00, float("inf"))
KINDS = ("translation", "similarity")
ANCHORS = ("last", "velocity")


@dataclass(frozen=True)
class MethodConfig:
    name: str
    kind: str
    anchor: str
    cap_m: float
    tangential_gate_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=("three", "dance", "box"),
        default=("three", "dance", "box"),
    )
    parser.add_argument("--max_cases_per_sequence", type=int, default=0)
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def finite_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,...j->...i", transform[:3, :3], points)
        + transform[:3, 3]
    )


def method_configs() -> list[MethodConfig]:
    output = []
    for anchor in ANCHORS:
        for kind in KINDS:
            output.append(
                MethodConfig(
                    name=f"ray_{anchor}_{kind}_uncapped",
                    kind=kind,
                    anchor=anchor,
                    cap_m=float("inf"),
                    tangential_gate_m=float("inf"),
                )
            )
            for cap in CAPS_M:
                for gate in GATES_M:
                    gate_name = "inf" if math.isinf(gate) else f"{gate:.2f}"
                    output.append(
                        MethodConfig(
                            name=(
                                f"ray_{anchor}_{kind}_cap{cap:.2f}_"
                                f"gate{gate_name}"
                            ),
                            kind=kind,
                            anchor=anchor,
                            cap_m=float(cap),
                            tangential_gate_m=float(gate),
                        )
                    )
    return output


def report_rows(sequences: tuple[str, ...]) -> list[dict]:
    rows = []
    if "three" in sequences:
        rows.extend(json.loads(THREE_REPORT.read_text(encoding="utf-8"))["cases"])
    if any(sequence in sequences for sequence in ("dance", "box")):
        frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))["cases"]
        rows.extend(row for row in frozen if row["sequence"] in sequences)
    return [row for row in rows if row["sequence"] in sequences]


def persistent_anchor(
    cache: dict, identity: str, mode: str
) -> tuple[np.ndarray, dict]:
    frames = [int(value) for value in cache["case"]["pre_frames"]]
    history = [
        (frame, humans[identity])
        for frame, humans in zip(frames, cache["humans"][:-1])
        if identity in humans
    ]
    if not history:
        raise KeyError(identity)
    history_frames = [frame for frame, _ in history]
    roots = [np.asarray(human["root"], dtype=np.float64) for _, human in history]
    delta = int(cache["case"]["post_frame"]) - history_frames[-1]
    velocity = geometry.robust_velocity(roots, history_frames)
    anchor = roots[-1] if mode == "last" else roots[-1] + delta * velocity
    return anchor, {
        "history": len(roots),
        "frame_delta": int(delta),
        "velocity_m_per_frame": velocity,
        "velocity_norm_m_per_frame": float(np.linalg.norm(velocity)),
    }


def point_errors(
    root: np.ndarray,
    joints: np.ndarray,
    vertices: np.ndarray,
    target_root: np.ndarray,
    target_joints: np.ndarray,
    target_vertices: np.ndarray,
) -> dict:
    joint_count = min(len(joints), len(target_joints))
    vertex_count = min(len(vertices), len(target_vertices))
    return {
        "root_error_m": float(np.linalg.norm(root - target_root)),
        "joint_error_m": float(
            np.linalg.norm(
                joints[:joint_count] - target_joints[:joint_count], axis=1
            ).mean()
        ),
        "vertex_error_m": float(
            np.linalg.norm(
                vertices[:vertex_count] - target_vertices[:vertex_count], axis=1
            ).mean()
        ),
    }


def apply_depth_change(
    root: np.ndarray,
    joints: np.ndarray,
    vertices: np.ndarray,
    camera_center: np.ndarray,
    ray: np.ndarray,
    depth_delta: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if kind == "translation":
        shift = depth_delta * ray
        return root + shift, joints + shift, vertices + shift, 1.0
    if kind != "similarity":
        raise KeyError(kind)
    current_depth = float(np.dot(root - camera_center, ray))
    scale = (current_depth + depth_delta) / max(current_depth, 1e-8)

    def scaled(points: np.ndarray) -> np.ndarray:
        return camera_center + scale * (points - camera_center)

    return scaled(root), scaled(joints), scaled(vertices), float(scale)


def pairwise_metrics(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict:
    identities = sorted(set(predicted) & set(target))
    distance_errors, vector_errors = [], []
    for index, first in enumerate(identities):
        for second in identities[index + 1 :]:
            predicted_vector = predicted[first] - predicted[second]
            target_vector = target[first] - target[second]
            distance_errors.append(
                abs(np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector))
            )
            vector_errors.append(np.linalg.norm(predicted_vector - target_vector))
    return {
        "pairwise_distance_error_m": (
            float(np.mean(distance_errors)) if distance_errors else float("nan")
        ),
        "pairwise_vector_error_m": (
            float(np.mean(vector_errors)) if vector_errors else float("nan")
        ),
    }


def evaluate_case(row: dict, configs: list[MethodConfig]) -> dict:
    sequence = str(row["sequence"])
    cache_path = SEQUENCE_INPUTS[sequence]["cache"] / f"{row['case']['key']}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    identities = tuple(
        identity
        for identity in geometry.SEQUENCE_IDENTITIES[sequence]
        if identity in cache["humans"][-1]
        and identity in cache["gt"]["post_humans"]
        and any(identity in frame for frame in cache["humans"][:-1])
    )
    boundary = np.asarray(row["methods"]["da3_safe"]["boundary"], dtype=np.float64)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    final_camera = boundary @ post_pose
    target_camera = gauge @ gt_post
    camera_center = final_camera[:3, 3]
    gt_camera_center = target_camera[:3, 3]

    method_rows: defaultdict[str, list[dict]] = defaultdict(list)
    roots_by_method: defaultdict[str, dict[str, np.ndarray]] = defaultdict(dict)
    target_roots: dict[str, np.ndarray] = {}
    person_diagnostics = []
    configs_by_anchor = {
        anchor: [config for config in configs if config.anchor == anchor]
        for anchor in ANCHORS
    }

    for identity in identities:
        predicted = cache["humans"][-1][identity]
        target = cache["gt"]["post_humans"][identity]
        root = transform_points(boundary, np.asarray(predicted["root"])[None])[0]
        joints = transform_points(boundary, np.asarray(predicted["joints"]))
        vertices = transform_points(boundary, np.asarray(predicted["vertices"]))
        target_root = transform_points(gauge, np.asarray(target["root"])[None])[0]
        target_joints = transform_points(gauge, np.asarray(target["joints"]))
        target_vertices = transform_points(gauge, np.asarray(target["vertices"]))
        target_roots[identity] = target_root

        ray_vector = root - camera_center
        ray_depth = float(np.linalg.norm(ray_vector))
        ray = ray_vector / max(ray_depth, 1e-8)
        error_vector = root - target_root
        signed_radial_error = float(np.dot(error_vector, ray))
        tangential_error = float(
            np.linalg.norm(error_vector - signed_radial_error * ray)
        )

        # Exact-camera decomposition isolates Human3R camera-local structure.
        gt_boundary = target_camera @ np.linalg.inv(post_pose)
        exact_root = transform_points(
            gt_boundary, np.asarray(predicted["root"])[None]
        )[0]
        exact_ray_vector = exact_root - gt_camera_center
        exact_ray = exact_ray_vector / max(float(np.linalg.norm(exact_ray_vector)), 1e-8)
        exact_error = exact_root - target_root
        exact_radial = float(np.dot(exact_error, exact_ray))
        exact_tangential = float(
            np.linalg.norm(exact_error - exact_radial * exact_ray)
        )

        current_errors = point_errors(
            root, joints, vertices, target_root, target_joints, target_vertices
        )
        method_rows["current_da3"].append(
            {
                **current_errors,
                "applied": False,
                "correction_m": 0.0,
                "scale": 1.0,
            }
        )
        roots_by_method["current_da3"][identity] = root

        # GT-ray oracle: maximum possible root gain from depth-only correction.
        oracle_depth = float(np.dot(target_root - camera_center, ray))
        oracle_delta = oracle_depth - ray_depth
        for kind in KINDS:
            corrected = apply_depth_change(
                root,
                joints,
                vertices,
                camera_center,
                ray,
                oracle_delta,
                kind,
            )
            method = f"oracle_gt_ray_{kind}"
            errors = point_errors(
                corrected[0],
                corrected[1],
                corrected[2],
                target_root,
                target_joints,
                target_vertices,
            )
            method_rows[method].append(
                {
                    **errors,
                    "applied": True,
                    "correction_m": abs(oracle_delta),
                    "scale": corrected[3],
                }
            )
            roots_by_method[method][identity] = corrected[0]

        anchor_debug = {}
        for anchor_mode in ANCHORS:
            anchor, motion = persistent_anchor(cache, identity, anchor_mode)
            anchor_depth = float(np.dot(anchor - camera_center, ray))
            projected_anchor = camera_center + anchor_depth * ray
            raw_delta = anchor_depth - ray_depth
            tangential_miss = float(np.linalg.norm(anchor - projected_anchor))
            anchor_debug[anchor_mode] = {
                **motion,
                "anchor": anchor,
                "raw_depth_correction_m": raw_delta,
                "anchor_tangential_miss_m": tangential_miss,
            }
            for config in configs_by_anchor[anchor_mode]:
                valid = bool(
                    anchor_depth > 0.05
                    and tangential_miss <= config.tangential_gate_m
                )
                depth_delta = (
                    float(np.clip(raw_delta, -config.cap_m, config.cap_m))
                    if valid
                    else 0.0
                )
                corrected = apply_depth_change(
                    root,
                    joints,
                    vertices,
                    camera_center,
                    ray,
                    depth_delta,
                    config.kind,
                )
                errors = point_errors(
                    corrected[0],
                    corrected[1],
                    corrected[2],
                    target_root,
                    target_joints,
                    target_vertices,
                )
                method_rows[config.name].append(
                    {
                        **errors,
                        "applied": bool(valid and abs(depth_delta) > 1e-9),
                        "correction_m": abs(depth_delta),
                        "scale": corrected[3],
                        "anchor_tangential_miss_m": tangential_miss,
                    }
                )
                roots_by_method[config.name][identity] = corrected[0]

        # Persistent 3D body extent: projection-preserving scale diagnostic.
        pre_extents = []
        for frame in cache["humans"][:-1]:
            if identity not in frame:
                continue
            human = frame[identity]
            centered = np.asarray(human["vertices"]) - np.asarray(human["root"])
            pre_extents.append(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))))
        current_centered = vertices - root
        post_extent = float(np.sqrt(np.mean(np.sum(current_centered**2, axis=1))))
        persistent_extent = float(np.median(pre_extents))
        extent_scale = float(np.clip(persistent_extent / max(post_extent, 1e-8), 0.8, 1.25))
        extent_delta = ray_depth * (extent_scale - 1.0)
        extent_corrected = apply_depth_change(
            root,
            joints,
            vertices,
            camera_center,
            ray,
            extent_delta,
            "similarity",
        )
        extent_errors = point_errors(
            extent_corrected[0],
            extent_corrected[1],
            extent_corrected[2],
            target_root,
            target_joints,
            target_vertices,
        )
        method_rows["persistent_mesh_extent_similarity"].append(
            {
                **extent_errors,
                "applied": bool(abs(extent_scale - 1.0) > 1e-9),
                "correction_m": abs(extent_delta),
                "scale": extent_scale,
            }
        )
        roots_by_method["persistent_mesh_extent_similarity"][identity] = extent_corrected[0]

        person_diagnostics.append(
            {
                "identity": identity,
                "current_root_error_m": current_errors["root_error_m"],
                "current_signed_radial_error_m": signed_radial_error,
                "current_abs_radial_error_m": abs(signed_radial_error),
                "current_tangential_error_m": tangential_error,
                "exact_camera_root_error_m": float(np.linalg.norm(exact_error)),
                "exact_camera_signed_radial_error_m": exact_radial,
                "exact_camera_abs_radial_error_m": abs(exact_radial),
                "exact_camera_tangential_error_m": exact_tangential,
                "ray_depth_m": ray_depth,
                "oracle_gt_ray_depth_correction_m": oracle_delta,
                "persistent_extent_scale": extent_scale,
                "anchors": anchor_debug,
            }
        )

    methods = {}
    for method, people in method_rows.items():
        methods[method] = {
            "people": people,
            **pairwise_metrics(roots_by_method[method], target_roots),
        }
    return {
        "sequence": sequence,
        "case": row["case"],
        "camera_span_deg": float(row["camera_span_deg"]),
        "da3_gate": row["proposal_diagnostics"]["da3_safe"],
        "camera": {
            "translation_error_m": float(
                np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3])
            ),
            "rotation_error_deg": float(
                row["methods"]["da3_safe"]["camera_rotation_error_deg"]
            ),
        },
        "person_diagnostics": person_diagnostics,
        "methods": methods,
    }


def summarize(cases: list[dict]) -> dict:
    method_names = sorted(set().union(*(case["methods"].keys() for case in cases)))
    output = {
        "case_count": len(cases),
        "human_count": int(
            sum(len(case["person_diagnostics"]) for case in cases)
        ),
        "methods": {},
    }
    current = [
        person
        for case in cases
        for person in case["methods"]["current_da3"]["people"]
    ]
    for method in method_names:
        people = [
            person for case in cases for person in case["methods"][method]["people"]
        ]
        row = {
            metric: finite_stats([person[metric] for person in people])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        deltas = [
            person["root_error_m"] - baseline["root_error_m"]
            for person, baseline in zip(people, current)
        ]
        row.update(
            {
                "root_mean_delta_m": float(np.mean(deltas)),
                "root_improvement_rate": float(np.mean(np.asarray(deltas) < 0.0)),
                "root_harm_over_5cm_rate": float(
                    np.mean(np.asarray(deltas) > 0.05)
                ),
                "coverage": float(np.mean([person["applied"] for person in people])),
                "correction_m": finite_stats(
                    [person["correction_m"] for person in people]
                ),
                "scale": finite_stats([person["scale"] for person in people]),
                "pairwise_distance_error_m": finite_stats(
                    [
                        case["methods"][method]["pairwise_distance_error_m"]
                        for case in cases
                    ]
                ),
                "pairwise_vector_error_m": finite_stats(
                    [
                        case["methods"][method]["pairwise_vector_error_m"]
                        for case in cases
                    ]
                ),
            }
        )
        output["methods"][method] = row

    diagnostics = [
        person for case in cases for person in case["person_diagnostics"]
    ]
    output["error_decomposition"] = {
        metric: finite_stats([person[metric] for person in diagnostics])
        for metric in (
            "current_root_error_m",
            "current_signed_radial_error_m",
            "current_abs_radial_error_m",
            "current_tangential_error_m",
            "exact_camera_root_error_m",
            "exact_camera_signed_radial_error_m",
            "exact_camera_abs_radial_error_m",
            "exact_camera_tangential_error_m",
            "oracle_gt_ray_depth_correction_m",
            "persistent_extent_scale",
        )
    }
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 B0 + DA3 Per-Person Ray-Anchor Feasibility",
        "",
        "The camera Boundary is frozen. GT identity/geometry are evaluation-only.",
        "",
        "## Error decomposition",
        "",
        "| Split | Humans | DA3 root | DA3 tangential | Exact-camera root | Exact-camera tangential |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("three", "dance", "box", "overall"):
        summary = report["by_sequence"].get(split) if split != "overall" else report["summary"]
        if summary is None:
            continue
        decomposition = summary["error_decomposition"]
        lines.append(
            f"| {split} | {summary['human_count']} | "
            f"{decomposition['current_root_error_m']['mean']:.4f} | "
            f"{decomposition['current_tangential_error_m']['mean']:.4f} | "
            f"{decomposition['exact_camera_root_error_m']['mean']:.4f} | "
            f"{decomposition['exact_camera_tangential_error_m']['mean']:.4f} |"
        )

    methods = report["summary"]["methods"]
    ranked = sorted(
        (
            name
            for name in methods
            if not name.startswith("oracle_") and name != "current_da3"
        ),
        key=lambda name: methods[name]["root_error_m"]["mean"],
    )
    shown = ["current_da3"] + ranked[:20] + [
        "oracle_gt_ray_translation",
        "oracle_gt_ray_similarity",
    ]
    lines.extend(
        [
            "",
            "## Overall ranked methods",
            "",
            "| Method | Root | Joint | Vertex | Delta | Improve | Harm >5cm | Coverage | Pair vector |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in shown:
        value = methods[name]
        lines.append(
            f"| {name} | {value['root_error_m']['mean']:.4f} | "
            f"{value['joint_error_m']['mean']:.4f} | "
            f"{value['vertex_error_m']['mean']:.4f} | "
            f"{value['root_mean_delta_m']:+.4f} | "
            f"{value['root_improvement_rate']:.1%} | "
            f"{value['root_harm_over_5cm_rate']:.1%} | "
            f"{value['coverage']:.1%} | "
            f"{value['pairwise_vector_error_m']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`translation` preserves body scale; `similarity` scales the body about the "
            "new camera center and therefore preserves every 2D projected vertex.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sequences = tuple(args.sequences)
    configs = method_configs()
    selected = []
    counts: defaultdict[str, int] = defaultdict(int)
    for row in report_rows(sequences):
        sequence = str(row["sequence"])
        if (
            args.max_cases_per_sequence
            and counts[sequence] >= args.max_cases_per_sequence
        ):
            continue
        selected.append(row)
        counts[sequence] += 1

    cases = []
    for index, row in enumerate(selected, start=1):
        case = evaluate_case(row, configs)
        cases.append(case)
        current = case["methods"]["current_da3"]["people"]
        print(
            f"[{index:03d}/{len(selected):03d}] {row['case']['key']} "
            f"humans={len(current)} root={np.mean([x['root_error_m'] for x in current]):.4f}",
            flush=True,
        )

    report = {
        "experiment": "v14_b0_da3_person_ray_anchor",
        "protocol": {
            "camera": "frozen B0 + frozen DA3-safe Boundary; never changed",
            "human": (
                "per-person first-post root ray plus persistent pre-cut Human3R "
                "trajectory; translation or camera-centered similarity"
            ),
            "identity": "cached GT mesh assignment; feasibility/evaluation only",
            "future_post_frames": False,
            "gt_candidate_generation": False,
            "oracle_methods": "oracle_gt_ray_* only",
        },
        "parameters": {
            "caps_m": CAPS_M,
            "tangential_gates_m": GATES_M,
            "anchor_modes": ANCHORS,
            "kinds": KINDS,
        },
        "summary": summarize(cases),
        "by_sequence": {
            sequence: summarize(
                [case for case in cases if case["sequence"] == sequence]
            )
            for sequence in sequences
        },
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "v14_b0_da3_person_ray_anchor.json"
    md_path = args.output_dir / "v14_b0_da3_person_ray_anchor.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)
    print(f">> JSON: {json_path}", flush=True)
    print(f">> Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
