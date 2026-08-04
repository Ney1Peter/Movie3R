#!/usr/bin/env python3
"""Causal, camera-safe within-shot root-stability diagnostics and C1 scan.

The program consumes only caches created by
``cache_streaming_within_shot_sequence.py``.  Runtime policies use predicted
camera, BRTC-LC geometry, and their own per-track history only.  GT arrays are
opened by the evaluator after a candidate trajectory is committed.

All C1 policies are deliberately small, fixed-state baselines rather than a
new pretrained component:

    camera-local root/body motion -> static/moving hysteresis gate
    -> causal root filter only while static -> one rigid world translation

Moving/unknown tracks receive an exact B0+BRTC fallback.  Cameras do not even
enter the output path, so the camera-safety invariant is structural.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/within_shot_stability/c1_static_gate"


@dataclass(frozen=True)
class Policy:
    name: str
    filter_kind: str
    alpha: float
    median_window: int
    warmup: int
    root_enter_m: float
    root_exit_m: float
    body_enter_m: float
    body_exit_m: float
    moving_hold_frames: int
    correction_cap_m: float


# These are fixed before reading this phase's evaluation results.  The two
# gates correspond to 1 cm/frame and 6 mm/frame camera-local root tolerance;
# 2.5 cm / 2 cm is an explicit moving exit threshold, not a GT-fitted value.
POLICIES = (
    Policy("c1_ema25", "ema", 0.25, 1, 2, 0.010, 0.025, 0.015, 0.035, 3, 0.15),
    Policy("c1_ema50", "ema", 0.50, 1, 2, 0.010, 0.025, 0.015, 0.035, 3, 0.15),
    Policy("c1_median3", "median", 1.00, 3, 2, 0.010, 0.025, 0.015, 0.035, 3, 0.15),
    Policy("c1_hold", "hold", 0.00, 1, 2, 0.010, 0.025, 0.015, 0.035, 3, 0.15),
    Policy("c1_conservative_ema25", "ema", 0.25, 1, 3, 0.006, 0.020, 0.010, 0.025, 4, 0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnose", "scan", "self-test"), required=True)
    parser.add_argument("--caches", type=Path, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--static-net-displacement-m", type=float, default=0.05,
        help="Evaluator-only GT label: net root displacement <= this is static",
    )
    parser.add_argument(
        "--moving-net-displacement-m", type=float, default=0.10,
        help="Evaluator-only GT label: net root displacement >= this is moving",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "b0_cameras", "b0_roots", "b0_joints", "b0_vertices",
            "brtc_shifts_by_track", "native_ids", "gt_cameras_evaluator_only",
            "gt_roots_evaluator_only", "gt_joints_evaluator_only", "gt_vertices_evaluator_only",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} misses fields {sorted(missing)}")
        result = {key: np.asarray(payload[key], dtype=np.float64) for key in required}
    if result["b0_roots"].ndim != 3 or result["b0_roots"].shape[-1] != 3:
        raise ValueError(f"Invalid root shape in {path}: {result['b0_roots'].shape}")
    return result


def camera_local(cameras: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Cameras are c2w; points may be (T,N,3) or (T,N,J,3)."""
    rotation = cameras[:, :3, :3]
    translation = cameras[:, :3, 3]
    translation_shape = (len(points),) + (1,) * (points.ndim - 2) + (3,)
    centered = points - translation.reshape(translation_shape)
    return np.einsum("tji,tn...j->tn...i", rotation, centered)


def clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value if norm <= limit else value * (float(limit) / max(norm, 1e-12))


def body_change(local_joints: np.ndarray) -> np.ndarray:
    """Per-frame body deformation after removing the local root translation."""
    centered = local_joints - local_joints[:, :, :1]
    delta = np.linalg.norm(centered[1:] - centered[:-1], axis=3).mean(axis=2)
    return np.vstack((np.zeros((1, centered.shape[1]), dtype=np.float64), delta))


def runtime_c1(cache: dict[str, np.ndarray], policy: Policy) -> dict[str, Any]:
    """Run exactly once in arrival order.  No evaluator field is referenced."""
    cameras = cache["b0_cameras"]
    b0_roots = cache["b0_roots"]
    shifts = cache["brtc_shifts_by_track"]
    roots = b0_roots + shifts[None]
    joints = cache["b0_joints"] + shifts[None, :, None, :]
    local_roots = camera_local(cameras, roots)
    local_joints = camera_local(cameras, joints)
    root_step = np.vstack((np.zeros((1, roots.shape[1])), np.linalg.norm(local_roots[1:] - local_roots[:-1], axis=2)))
    body_step = body_change(local_joints)
    frame_count, person_count = roots.shape[:2]
    output = roots.copy()
    residuals = np.zeros_like(roots)
    gates = np.zeros((frame_count, person_count), dtype=np.int8)  # 1 only when filtered static
    reasons = np.full((frame_count, person_count), "warmup", dtype="U16")

    for person in range(person_count):
        filtered: np.ndarray | None = None
        local_history: list[np.ndarray] = []
        moving_until = -1
        for frame in range(frame_count):
            observation = local_roots[frame, person]
            local_history.append(observation.copy())
            if frame < int(policy.warmup):
                filtered = observation.copy()
                reasons[frame, person] = "warmup"
                continue
            recent_root = root_step[max(1, frame - 2): frame + 1, person]
            recent_body = body_step[max(1, frame - 2): frame + 1, person]
            stable_evidence = (
                float(np.median(recent_root)) <= float(policy.root_enter_m)
                and float(np.median(recent_body)) <= float(policy.body_enter_m)
            )
            moving_evidence = (
                float(root_step[frame, person]) > float(policy.root_exit_m)
                or float(body_step[frame, person]) > float(policy.body_exit_m)
            )
            if moving_evidence:
                moving_until = frame + int(policy.moving_hold_frames)
            is_static = bool(stable_evidence and frame > moving_until)
            if not is_static:
                filtered = observation.copy()
                reasons[frame, person] = "moving" if frame <= moving_until else "unknown"
                continue
            if filtered is None:
                filtered = observation.copy()
            if policy.filter_kind == "ema":
                filtered = float(policy.alpha) * observation + (1.0 - float(policy.alpha)) * filtered
            elif policy.filter_kind == "median":
                filtered = np.median(np.stack(local_history[-int(policy.median_window):]), axis=0)
            elif policy.filter_kind == "hold":
                filtered = filtered.copy()
            else:
                raise ValueError(f"Unknown filter {policy.filter_kind}")
            correction_local = clip_norm(filtered - observation, float(policy.correction_cap_m))
            # local = R^T (world - camera); turn the root-only residual back into world.
            correction_world = cameras[frame, :3, :3] @ correction_local
            output[frame, person] = roots[frame, person] + correction_world
            residuals[frame, person] = correction_world
            gates[frame, person] = 1
            reasons[frame, person] = "static"
    return {
        "roots": output,
        "residuals": residuals,
        "gates": gates,
        "reasons": reasons,
        "runtime_features": {"local_roots": local_roots, "root_step": root_step, "body_step": body_step},
        "camera_max_abs_change": 0.0,
    }


def trajectory_metrics(roots: np.ndarray, cameras: np.ndarray) -> dict[str, np.ndarray]:
    local = camera_local(cameras, roots)
    step_world = np.linalg.norm(roots[1:] - roots[:-1], axis=2)
    step_local = np.linalg.norm(local[1:] - local[:-1], axis=2)
    from_start_world = np.linalg.norm(roots - roots[:1], axis=2)
    from_start_local = np.linalg.norm(local - local[:1], axis=2)
    acceleration = np.linalg.norm(roots[2:] - 2.0 * roots[1:-1] + roots[:-2], axis=2)
    return {
        "world_net_displacement_m": from_start_world[-1],
        "world_max_deviation_m": from_start_world.max(axis=0),
        "world_path_length_m": step_world.sum(axis=0),
        "camera_local_net_displacement_m": from_start_local[-1],
        "camera_local_max_deviation_m": from_start_local.max(axis=0),
        "camera_local_path_length_m": step_local.sum(axis=0),
        "root_acceleration_m_per_frame2": acceleration.mean(axis=0),
    }


def mean_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted - target, axis=-1)


def layout_vector_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean per-frame pairwise root-vector error, shape (T,)."""
    values = []
    for first in range(predicted.shape[1]):
        for second in range(first + 1, predicted.shape[1]):
            predicted_vector = predicted[:, first] - predicted[:, second]
            target_vector = target[:, first] - target[:, second]
            values.append(np.linalg.norm(predicted_vector - target_vector, axis=1))
    if not values:
        return np.zeros(predicted.shape[0], dtype=np.float64)
    return np.stack(values, axis=1).mean(axis=1)


def finite_mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def aggregate_labels(gt_roots: np.ndarray, static_limit: float, moving_limit: float) -> np.ndarray:
    net = np.linalg.norm(gt_roots[-1] - gt_roots[0], axis=1)
    labels = np.full(len(net), "ambiguous", dtype="U9")
    labels[net <= float(static_limit)] = "static"
    labels[net >= float(moving_limit)] = "moving"
    return labels


def evaluate_cache(
    path: Path,
    policy: Policy | None,
    static_limit: float,
    moving_limit: float,
) -> dict[str, Any]:
    cache = load_cache(path)
    baseline_roots = cache["b0_roots"] + cache["brtc_shifts_by_track"][None]
    target_roots = cache["gt_roots_evaluator_only"]
    labels = aggregate_labels(target_roots, static_limit, moving_limit)
    baseline_metrics = trajectory_metrics(baseline_roots, cache["b0_cameras"])
    target_metrics = trajectory_metrics(target_roots, cache["gt_cameras_evaluator_only"])
    candidate = runtime_c1(cache, policy) if policy is not None else None
    candidate_roots = candidate["roots"] if candidate is not None else baseline_roots
    candidate_metrics = trajectory_metrics(candidate_roots, cache["b0_cameras"])
    candidate_root_error = mean_error(candidate_roots, target_roots)
    baseline_root_error = mean_error(baseline_roots, target_roots)
    baseline_layout_error = layout_vector_error(baseline_roots, target_roots)
    candidate_layout_error = layout_vector_error(candidate_roots, target_roots)

    # A candidate translation is applied identically to root/joints/vertices.
    residuals = candidate["residuals"] if candidate is not None else np.zeros_like(baseline_roots)
    baseline_joints = cache["b0_joints"] + cache["brtc_shifts_by_track"][None, :, None, :]
    candidate_joints = baseline_joints + residuals[:, :, None, :]
    baseline_vertices = cache["b0_vertices"] + cache["brtc_shifts_by_track"][None, :, None, :]
    candidate_vertices = baseline_vertices + residuals[:, :, None, :]
    joint_count = min(baseline_joints.shape[2], cache["gt_joints_evaluator_only"].shape[2])
    baseline_joint_error = mean_error(
        baseline_joints[:, :, :joint_count], cache["gt_joints_evaluator_only"][:, :, :joint_count]
    ).mean(axis=2)
    candidate_joint_error = mean_error(
        candidate_joints[:, :, :joint_count], cache["gt_joints_evaluator_only"][:, :, :joint_count]
    ).mean(axis=2)
    baseline_vertex_error = mean_error(baseline_vertices, cache["gt_vertices_evaluator_only"]).mean(axis=2)
    candidate_vertex_error = mean_error(candidate_vertices, cache["gt_vertices_evaluator_only"]).mean(axis=2)

    people = []
    for person in range(baseline_roots.shape[1]):
        label = str(labels[person])
        baseline_net = float(baseline_metrics["world_net_displacement_m"][person])
        candidate_net = float(candidate_metrics["world_net_displacement_m"][person])
        people.append({
            "track": int(person), "label_evaluator_only": label,
            "gt": {key: float(value[person]) for key, value in target_metrics.items()},
            "baseline": {key: float(value[person]) for key, value in baseline_metrics.items()},
            "candidate": {key: float(value[person]) for key, value in candidate_metrics.items()},
            "root_error_m": {
                "baseline_mean": float(baseline_root_error[:, person].mean()),
                "candidate_mean": float(candidate_root_error[:, person].mean()),
                "baseline_final": float(baseline_root_error[-1, person]),
                "candidate_final": float(candidate_root_error[-1, person]),
            },
            "joint_error_m": {"baseline_mean": float(baseline_joint_error[:, person].mean()), "candidate_mean": float(candidate_joint_error[:, person].mean())},
            "vertex_error_m": {"baseline_mean": float(baseline_vertex_error[:, person].mean()), "candidate_mean": float(candidate_vertex_error[:, person].mean())},
            "moving_displacement_retention": (
                float(candidate_net / baseline_net) if baseline_net > 1e-8 else None
            ),
            "moving_path_retention": (
                float(candidate_metrics["world_path_length_m"][person] /
                      baseline_metrics["world_path_length_m"][person])
                if float(baseline_metrics["world_path_length_m"][person]) > 1e-8 else None
            ),
            "gate": (
                {"static_filtered_frames": int(candidate["gates"][:, person].sum()),
                 "filtered_fraction": float(candidate["gates"][:, person].mean()),
                 "reasons": candidate["reasons"][:, person].tolist(),
                 "max_residual_m": float(np.linalg.norm(residuals[:, person], axis=1).max())}
                if candidate is not None else None
            ),
        })

    static_rows = [item for item in people if item["label_evaluator_only"] == "static"]
    moving_rows = [item for item in people if item["label_evaluator_only"] == "moving"]
    summary = {
        "person_count": len(people),
        "static_count_evaluator_only": len(static_rows),
        "moving_count_evaluator_only": len(moving_rows),
        "static_camera_local_path_reduction": (
            1.0 - finite_mean(np.asarray([p["candidate"]["camera_local_path_length_m"] for p in static_rows])) /
            max(finite_mean(np.asarray([p["baseline"]["camera_local_path_length_m"] for p in static_rows])) or 0.0, 1e-12)
            if static_rows else None
        ),
        "static_camera_local_max_deviation_reduction": (
            1.0 - finite_mean(np.asarray([p["candidate"]["camera_local_max_deviation_m"] for p in static_rows])) /
            max(finite_mean(np.asarray([p["baseline"]["camera_local_max_deviation_m"] for p in static_rows])) or 0.0, 1e-12)
            if static_rows else None
        ),
        "static_root_error_relative_change": (
            finite_mean(np.asarray([p["root_error_m"]["candidate_mean"] for p in static_rows])) /
            max(finite_mean(np.asarray([p["root_error_m"]["baseline_mean"] for p in static_rows])) or 0.0, 1e-12) - 1.0
            if static_rows else None
        ),
        "moving_min_displacement_retention": (
            min(p["moving_displacement_retention"] for p in moving_rows if p["moving_displacement_retention"] is not None)
            if moving_rows else None
        ),
        "moving_min_path_retention": (
            min(p["moving_path_retention"] for p in moving_rows if p["moving_path_retention"] is not None)
            if moving_rows else None
        ),
        "all_root_error_relative_change": float(candidate_root_error.mean() / max(baseline_root_error.mean(), 1e-12) - 1.0),
        "all_joint_error_relative_change": float(candidate_joint_error.mean() / max(baseline_joint_error.mean(), 1e-12) - 1.0),
        "all_vertex_error_relative_change": float(candidate_vertex_error.mean() / max(baseline_vertex_error.mean(), 1e-12) - 1.0),
        "all_layout_vector_error_relative_change": float(
            candidate_layout_error.mean() / max(baseline_layout_error.mean(), 1e-12) - 1.0
        ),
        "baseline_layout_vector_error_m": float(baseline_layout_error.mean()),
        "candidate_layout_vector_error_m": float(candidate_layout_error.mean()),
        "root_harm_over_5cm_rate_change": float(
            np.mean(candidate_root_error - baseline_root_error > 0.05) - 0.0
        ),
        "camera_max_abs_change": float(candidate["camera_max_abs_change"] if candidate is not None else 0.0),
    }
    return {"cache": str(path), "policy": asdict(policy) if policy else None, "summary": summary, "people": people}


def markdown(report: dict[str, Any], title: str) -> str:
    lines = [f"# {title}", "", "GT is evaluator-only. C1 uses only arrived predicted geometry and leaves cameras bit-exact.", ""]
    lines.extend([
        "| Cache / policy | static | moving | static local-path reduction | static max-deviation reduction | static root change | min moving net/path retention | all root change | layout change | camera change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["results"]:
        summary = row["summary"]
        policy = row["policy"]["name"] if row["policy"] else "B0+BRTC diagnostic"
        pct = lambda value: "--" if value is None else f"{100.0 * value:+.1f}%"
        camera = f"{summary['camera_max_abs_change']:.1e}"
        lines.append(
            f"| {Path(row['cache']).stem} / {policy} | {summary['static_count_evaluator_only']} | "
            f"{summary['moving_count_evaluator_only']} | {pct(summary['static_camera_local_path_reduction'])} | "
            f"{pct(summary['static_camera_local_max_deviation_reduction'])} | "
            f"{pct(summary['static_root_error_relative_change'])} | "
            f"{pct(None if summary['moving_min_displacement_retention'] is None else summary['moving_min_displacement_retention'] - 1.0)} / "
            f"{pct(None if summary['moving_min_path_retention'] is None else summary['moving_min_path_retention'] - 1.0)} | "
            f"{pct(summary['all_root_error_relative_change'])} | "
            f"{pct(summary['all_layout_vector_error_relative_change'])} | {camera} |"
        )
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    cameras = np.repeat(np.eye(4)[None], 5, axis=0)
    roots = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.01, 0, 0], [0.04, 0, 0]], [[-0.01, 0, 0], [0.08, 0, 0]], [[0.01, 0, 0], [0.12, 0, 0]], [[-0.01, 0, 0], [0.16, 0, 0]]], dtype=np.float64)
    joints = np.repeat(roots[:, :, None, :], 4, axis=2)
    cache = {
        "b0_cameras": cameras, "b0_roots": roots, "b0_joints": joints,
        "brtc_shifts_by_track": np.zeros((2, 3)),
    }
    result = runtime_c1(cache, POLICIES[0])
    assert result["camera_max_abs_change"] == 0.0
    # The obvious walker is never filtered after its first high-motion frame.
    assert not result["gates"][2:, 1].any()
    # Prefix and full-stream inference must agree exactly (causality test).
    prefix = {key: value[:4] if key != "brtc_shifts_by_track" else value for key, value in cache.items()}
    short = runtime_c1(prefix, POLICIES[0])
    assert np.allclose(short["roots"], result["roots"][:4])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.phase == "self-test":
        self_test()
        return
    caches = [path.resolve() for path in args.caches]
    if not caches:
        raise ValueError("At least one --caches path is required")
    output = args.output_dir.resolve()
    if output != REPO_ROOT and REPO_ROOT not in output.parents:
        raise ValueError("Output must stay under Movie3R")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "causal": "frame t reads only its track state and predicted P0+B0+BRTC geometry at <=t",
        "camera": "structurally untouched; reported max change must be 0",
        "gt": "opened only by evaluate_cache after runtime_c1 returns",
        "static_moving_labels": {
            "static_net_displacement_m": float(args.static_net_displacement_m),
            "moving_net_displacement_m": float(args.moving_net_displacement_m),
        },
    }
    if args.phase == "diagnose":
        results = [evaluate_cache(path, None, args.static_net_displacement_m, args.moving_net_displacement_m) for path in caches]
        report = {"experiment": "within_shot_c0_diagnosis", "protocol": protocol, "results": results}
        (output / "C0_DIAGNOSIS.json").write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "C0_DIAGNOSIS.md").write_text(markdown(report, "Within-shot C0 diagnosis"), encoding="utf-8")
        print(markdown(report, "Within-shot C0 diagnosis"))
        return
    # All policies were serialized in this source before the scan.  This scan
    # reports every one; it does not use GT inside their gate or filter.
    results = []
    for policy in POLICIES:
        for path in caches:
            results.append(evaluate_cache(path, policy, args.static_net_displacement_m, args.moving_net_displacement_m))
    report = {
        "experiment": "within_shot_c1_fixed_policy_scan",
        "protocol": protocol,
        "policies_declared_before_scan": [asdict(policy) for policy in POLICIES],
        "policy_schema_sha256": canonical_hash([asdict(policy) for policy in POLICIES]),
        "results": results,
    }
    (output / "C1_FIXED_POLICY_SCAN.json").write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "C1_FIXED_POLICY_SCAN.md").write_text(markdown(report, "Within-shot C1 fixed-policy scan"), encoding="utf-8")
    print(markdown(report, "Within-shot C1 fixed-policy scan"))


if __name__ == "__main__":
    main()
