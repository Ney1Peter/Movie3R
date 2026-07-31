#!/usr/bin/env python3
"""Measure which causal cues explain the signed residual around frozen V14 B0.

This is a diagnostic probe. GT camera and GT meshes are used only to build targets and
evaluate proposals. Every proposal is generated from cached Human3R outputs, predicted
humans, predicted pointmaps, and the frozen learned B0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import (  # noqa: E402
    b0_human_candidates,
    seam_metrics,
    solution,
)


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/residual_observability"
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=tuple(SEQUENCE_INPUTS),
        default=("three",),
        help="Use three for development. Do not inspect dance/box while tuning a method.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--skip_identity_rebuild", action="store_true")
    parser.add_argument("--skip_scene_refinement", action="store_true")
    return parser.parse_args()


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = np.asarray(rotation, dtype=np.float64)
    output[:3, 3] = np.asarray(translation, dtype=np.float64)
    return output


def vector_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
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


def gt_boundary(cache: dict) -> np.ndarray:
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post
    return target_camera @ np.linalg.inv(post_pose)


def right_residual(base: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.linalg.inv(np.asarray(base, dtype=np.float64)) @ np.asarray(
        target, dtype=np.float64
    )
    rotvec = Rotation.from_matrix(residual[:3, :3]).as_rotvec()
    return rotvec, residual[:3, 3]


def bounded_vector(value: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value * min(1.0, float(maximum) / max(norm, 1e-12))


def evaluate_boundary(cache: dict, boundary: np.ndarray, identities) -> dict:
    return geometry.evaluate_solution(
        cache,
        solution(boundary[:3, :3], boundary[:3, 3], identities),
    )


def mutual_scene_cue(
    b0: np.ndarray, post_cloud: np.ndarray, pre_cloud: np.ndarray
) -> dict:
    post_cloud = np.asarray(post_cloud, dtype=np.float64)
    pre_cloud = np.asarray(pre_cloud, dtype=np.float64)
    if len(post_cloud) < 8 or len(pre_cloud) < 8:
        return {"status": "too_few_points"}
    mapped = geometry.transform_points(b0, post_cloud)
    pre_tree = cKDTree(pre_cloud)
    post_tree = cKDTree(mapped)
    distance, nearest_pre = pre_tree.query(mapped, k=1, workers=-1)
    _, nearest_post = post_tree.query(pre_cloud, k=1, workers=-1)
    post_index = np.arange(len(mapped), dtype=np.int64)
    mutual = nearest_post[nearest_pre] == post_index
    finite = np.isfinite(distance)
    keep = mutual & finite & (distance <= 0.60)
    if not np.any(keep):
        return {
            "status": "no_mutual_matches",
            "post_points": int(len(post_cloud)),
            "pre_points": int(len(pre_cloud)),
        }
    world_vectors = pre_cloud[nearest_pre[keep]] - mapped[keep]
    local_vectors = world_vectors @ b0[:3, :3]
    kept_distance = distance[keep]
    near = kept_distance <= 0.30
    selected = local_vectors[near] if int(np.sum(near)) >= 3 else local_vectors
    selected_distance = kept_distance[near] if int(np.sum(near)) >= 3 else kept_distance
    median = np.median(selected, axis=0)
    mad = np.median(np.abs(selected - median), axis=0)
    return {
        "status": "ok",
        "post_points": int(len(post_cloud)),
        "pre_points": int(len(pre_cloud)),
        "mutual_count_060": int(np.sum(keep)),
        "mutual_count_030": int(np.sum(keep & (distance <= 0.30))),
        "mutual_ratio_060": float(np.mean(mutual & finite & (distance <= 0.60))),
        "distance_median_m": float(np.median(selected_distance)),
        "distance_p90_m": float(np.percentile(selected_distance, 90)),
        "du_median": median,
        "du_mad": mad,
    }


def human_cues(cache: dict, b0: np.ndarray) -> tuple[dict, tuple[str, ...]]:
    candidates = b0_human_candidates(cache, b0)
    identities = tuple(sorted(candidates))
    if not identities:
        return {"status": "no_shared_humans", "count": 0}, identities
    root_du = []
    torso_rotvec = []
    quality = []
    torso_mad = []
    velocity = []
    for identity in identities:
        candidate = candidates[identity]
        aligned_root = geometry.transform_points(
            b0, np.asarray(candidate["post_root"])[None]
        )[0]
        world_root_delta = np.asarray(candidate["anchor"]) - aligned_root
        root_du.append(b0[:3, :3].T @ world_root_delta)
        relative = b0[:3, :3].T @ np.asarray(candidate["rotation"])
        torso_rotvec.append(Rotation.from_matrix(relative).as_rotvec())
        quality.append(float(candidate["quality"]))
        torso_mad.append(
            float(candidate["v16_debug"].get("angle_median_abs_deviation_deg", 0.0))
        )
        velocity.append(float(np.linalg.norm(candidate["root_velocity_m_per_frame"])))
    root_du_array = np.stack(root_du)
    torso_array = np.stack(torso_rotvec)
    weights = np.asarray(quality, dtype=np.float64)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    return {
        "status": "ok",
        "count": len(identities),
        "identities": identities,
        "root_du_mean": np.mean(root_du_array, axis=0),
        "root_du_median": np.median(root_du_array, axis=0),
        "root_du_weighted": np.average(root_du_array, axis=0, weights=weights),
        "root_du_dispersion_m": float(
            np.mean(np.linalg.norm(root_du_array - np.median(root_du_array, axis=0), axis=1))
        ),
        "torso_rotvec_mean": np.mean(torso_array, axis=0),
        "torso_rotvec_median": np.median(torso_array, axis=0),
        "torso_rotvec_weighted": np.average(torso_array, axis=0, weights=weights),
        "torso_rotation_dispersion_deg": float(
            np.degrees(
                np.mean(
                    np.linalg.norm(
                        torso_array - np.median(torso_array, axis=0), axis=1
                    )
                )
            )
        ),
        "quality_mean": float(np.mean(quality)),
        "quality_min": float(np.min(quality)),
        "torso_mad_deg_mean": float(np.mean(torso_mad)),
        "root_velocity_m_per_frame_mean": float(np.mean(velocity)),
    }, identities


def flatten_features(row: dict) -> dict[str, float]:
    features = {
        "camera_span_deg": float(row["camera_span_deg"]),
        "b0_rotation_deg": float(row["b0_rotation_deg"]),
        "b0_translation_norm_m": float(row["b0_translation_norm_m"]),
        "human_count": float(row["human"].get("count", 0)),
        "root_du_dispersion_m": float(
            row["human"].get("root_du_dispersion_m", float("nan"))
        ),
        "torso_rotation_dispersion_deg": float(
            row["human"].get("torso_rotation_dispersion_deg", float("nan"))
        ),
        "human_quality_mean": float(
            row["human"].get("quality_mean", float("nan"))
        ),
        "human_quality_min": float(row["human"].get("quality_min", float("nan"))),
        "root_velocity_mean": float(
            row["human"].get("root_velocity_m_per_frame_mean", float("nan"))
        ),
        "scene_mutual_ratio": float(
            row["scene"].get("mutual_ratio_060", float("nan"))
        ),
        "scene_distance_median": float(
            row["scene"].get("distance_median_m", float("nan"))
        ),
        "scene_distance_p90": float(
            row["scene"].get("distance_p90_m", float("nan"))
        ),
        "scene_nn_median": float(row["seam"]["background_cloud_nn_median_m"]),
        "root_jump_mean": float(row["seam"]["root_jump_mean_m"]),
    }
    vector_features = {
        "human_root_mean": row["human"].get("root_du_mean"),
        "human_root_median": row["human"].get("root_du_median"),
        "human_root_weighted": row["human"].get("root_du_weighted"),
        "human_torso_mean": row["human"].get("torso_rotvec_mean"),
        "human_torso_median": row["human"].get("torso_rotvec_median"),
        "human_torso_weighted": row["human"].get("torso_rotvec_weighted"),
        "scene_du_median": row["scene"].get("du_median"),
        "scene_icp_rotvec": row["scene_refinement"].get("rotvec"),
        "scene_icp_du": row["scene_refinement"].get("du"),
    }
    for prefix, value in vector_features.items():
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64).reshape(3)
        for axis, item in zip(AXES, array):
            features[f"{prefix}_{axis}"] = float(item)
    return features


def correlation(first: np.ndarray, second: np.ndarray) -> dict:
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < 4 or float(np.std(first)) < 1e-12 or float(np.std(second)) < 1e-12:
        return {"count": int(len(first)), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "count": int(len(first)),
        "pearson": float(pearsonr(first, second).statistic),
        "spearman": float(spearmanr(first, second).statistic),
    }


def analyze(rows: list[dict]) -> dict:
    targets = {}
    for axis_index, axis in enumerate(AXES):
        targets[f"gt_du_{axis}"] = np.asarray(
            [row["gt_right_du"][axis_index] for row in rows], dtype=np.float64
        )
        targets[f"gt_rotvec_{axis}"] = np.asarray(
            [row["gt_right_rotvec"][axis_index] for row in rows], dtype=np.float64
        )
    features_by_row = [flatten_features(row) for row in rows]
    feature_names = sorted(set().union(*(item.keys() for item in features_by_row)))
    correlations = {}
    for feature in feature_names:
        values = np.asarray(
            [item.get(feature, float("nan")) for item in features_by_row], dtype=np.float64
        )
        correlations[feature] = {
            target: correlation(values, target_values)
            for target, target_values in targets.items()
        }

    method_names = sorted(set().union(*(set(row["methods"]) for row in rows)))
    method_summary = {}
    for method in method_names:
        method_rows = [row for row in rows if method in row["methods"]]
        baseline_rows = [row for row in method_rows if "b0" in row["methods"]]
        values = {
            metric: vector_stats([row["methods"][method][metric] for row in method_rows])
            for metric in (
                "camera_translation_error_m",
                "camera_rotation_error_deg",
                "camera_composite",
                "human_root_error_m",
                "pairwise_distance_error_m",
            )
        }
        values["valid_cases"] = len(method_rows)
        values["paired_against_b0"] = {
            metric: {
                "mean_delta": float(
                    np.mean(
                        [
                            row["methods"][method][metric]
                            - row["methods"]["b0"][metric]
                            for row in baseline_rows
                        ]
                    )
                ),
                "improvement_rate": float(
                    np.mean(
                        [
                            row["methods"][method][metric]
                            < row["methods"]["b0"][metric]
                            for row in baseline_rows
                        ]
                    )
                ),
            }
            for metric in ("camera_composite", "human_root_error_m")
        }
        method_summary[method] = values
    return {
        "case_count": len(rows),
        "gt_residual": {
            "rotation_deg": vector_stats(
                [math.degrees(np.linalg.norm(row["gt_right_rotvec"])) for row in rows]
            ),
            "translation_m": vector_stats(
                [np.linalg.norm(row["gt_right_du"]) for row in rows]
            ),
            **{
                f"du_{axis}": vector_stats([row["gt_right_du"][index] for row in rows])
                for index, axis in enumerate(AXES)
            },
        },
        "methods": method_summary,
        "correlations": correlations,
    }


def load_case(
    sequence: str,
    report_case: dict,
    args: argparse.Namespace,
) -> dict:
    inputs = SEQUENCE_INPUTS[sequence]
    cache_path = inputs["cache"] / f"{report_case['case']['key']}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    if not args.skip_identity_rebuild:
        cache = geometry.reassign_cache_gt_identities(
            SimpleNamespace(data_root=args.data_root, size=int(args.size), sequence=sequence),
            cache,
        )
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    target = gt_boundary(cache)
    gt_rotvec, gt_du = right_residual(b0, target)
    human, identities = human_cues(cache, b0)
    if not identities:
        identities = tuple(
            identity
            for identity in geometry.IDENTITIES
            if identity in cache["humans"][-1]
        )
    pre_clouds = [
        np.asarray(value, dtype=np.float64)
        for value in cache["clouds"][:-1]
        if len(value)
    ]
    pre_cloud = (
        np.concatenate(pre_clouds)
        if pre_clouds
        else np.empty((0, 3), dtype=np.float64)
    )
    post_cloud = np.asarray(cache["clouds"][-1], dtype=np.float64)
    scene = mutual_scene_cue(b0, post_cloud, pre_cloud)
    scene_refinement = {"status": "skipped"}
    refined = None
    if not args.skip_scene_refinement and len(pre_cloud) >= 32 and len(post_cloud) >= 32:
        refined, debug = geometry.fixed_refine(b0, post_cloud, pre_cloud)
        rotvec, du = right_residual(b0, refined)
        scene_refinement = {
            "status": debug.get("status", "ok"),
            "rotvec": rotvec,
            "du": du,
            "debug": debug,
        }

    identity_residual = np.eye(4, dtype=np.float64)
    rotation_only = transform(Rotation.from_rotvec(gt_rotvec).as_matrix(), np.zeros(3))
    translation_only = transform(np.eye(3), gt_du)
    methods = {
        "b0": evaluate_boundary(cache, b0, identities),
        "oracle_full": evaluate_boundary(cache, target, identities),
        "oracle_rotation_only": evaluate_boundary(cache, b0 @ rotation_only, identities),
        "oracle_translation_only": evaluate_boundary(cache, b0 @ translation_only, identities),
    }
    for index, axis in enumerate(AXES):
        component = np.zeros(3, dtype=np.float64)
        component[index] = gt_du[index]
        methods[f"oracle_translation_{axis}"] = evaluate_boundary(
            cache, b0 @ transform(np.eye(3), component), identities
        )
    if refined is not None:
        methods["scene_icp_full"] = evaluate_boundary(cache, refined, identities)
    if scene.get("status") == "ok":
        scene_du = bounded_vector(np.asarray(scene["du_median"]), 0.25)
        methods["scene_mutual_translation_b025"] = evaluate_boundary(
            cache,
            b0 @ transform(identity_residual[:3, :3], scene_du),
            identities,
        )
    if human.get("status") == "ok":
        torso = bounded_vector(np.asarray(human["torso_rotvec_median"]), math.radians(20.0))
        methods["human_torso_rotation_b20"] = evaluate_boundary(
            cache,
            b0 @ transform(Rotation.from_rotvec(torso).as_matrix(), np.zeros(3)),
            identities,
        )

    return {
        "sequence": sequence,
        "case": report_case["case"],
        "camera_span_deg": float(report_case["camera_span_deg"]),
        "b0_rotation_deg": float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(b0[:3, :3]).as_rotvec()))
        ),
        "b0_translation_norm_m": float(np.linalg.norm(b0[:3, 3])),
        "gt_boundary": target,
        "gt_right_rotvec": gt_rotvec,
        "gt_right_du": gt_du,
        "human": human,
        "scene": scene,
        "scene_refinement": scene_refinement,
        "seam": seam_metrics(cache, b0),
        "methods": methods,
    }


def write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    methods = summary["methods"]
    lines = [
        "# V14 B0 Residual Observability",
        "",
        f"Sequences: `{', '.join(report['sequences'])}`. Cases: `{summary['case_count']}`.",
        "",
        "GT is used only for signed residual targets and evaluation.",
        "",
        "## Residual Headroom",
        "",
        "| Quantity | Mean | Median | P90 | P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("rotation_deg", "translation_m", "du_x", "du_y", "du_z"):
        stats = summary["gt_residual"][name]
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['p90']:.4f} | {stats['p95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Direct and Oracle Proposals",
            "",
            "| Method | N | Camera T | Camera R | Composite | Human root | Layout |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in methods.items():
        lines.append(
            f"| {method} | {values['valid_cases']} | "
            f"{values['camera_translation_error_m']['mean']:.4f} | "
            f"{values['camera_rotation_error_deg']['mean']:.3f} | "
            f"{values['camera_composite']['mean']:.4f} | "
            f"{values['human_root_error_m']['mean']:.4f} | "
            f"{values['pairwise_distance_error_m']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Signed-Cue Correlations",
            "",
            "Absolute Spearman values are listed only as diagnostics. A cue must still "
            "generalize and improve a frozen proposal.",
            "",
            "| Target | Feature | Spearman | Count |",
            "|---|---|---:|---:|",
        ]
    )
    targets = [f"gt_du_{axis}" for axis in AXES] + [f"gt_rotvec_{axis}" for axis in AXES]
    for target_name in targets:
        ranked = []
        for feature, target_rows in summary["correlations"].items():
            result = target_rows[target_name]
            if np.isfinite(result["spearman"]):
                ranked.append((abs(result["spearman"]), feature, result))
        for _, feature, result in sorted(ranked, reverse=True)[:5]:
            lines.append(
                f"| {target_name} | {feature} | {result['spearman']:.3f} | "
                f"{result['count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in args.sequences:
        report = json.loads(SEQUENCE_INPUTS[sequence]["report"].read_text())
        report_cases = report["cases"][: args.max_cases or None]
        for index, report_case in enumerate(report_cases, start=1):
            row = load_case(sequence, report_case, args)
            rows.append(row)
            print(
                f"[{sequence} {index:03d}/{len(report_cases):03d}] "
                f"{report_case['case']['key']}",
                flush=True,
            )
    report = {
        "experiment": "v14_b0_residual_observability",
        "sequences": list(args.sequences),
        "protocol": {
            "b0": "frozen learned B0 from existing identity reports",
            "identity": (
                "strict GT mesh-projection rebuild for WHERE isolation"
                if not args.skip_identity_rebuild
                else "existing cached labels"
            ),
            "gt_usage": "signed residual target and evaluation only",
            "scene_matches": "mutual nearest predicted background pointmap points after B0",
        },
        "summary": analyze(rows),
        "cases": rows,
    }
    json_path = args.output_dir / "v14_b0_residual_observability.json"
    md_path = args.output_dir / "v14_b0_residual_observability.md"
    json_path.write_text(json.dumps(serializable(report), indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
