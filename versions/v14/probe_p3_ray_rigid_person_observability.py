#!/usr/bin/env python3
"""P3: test whether B0-frozen two-view joint rays support a new rigid person action.

This is a CPU-only observability diagnostic.  All ray targets and both candidate
actions are constructed from serialized runtime fields before evaluator GT and
identity labels are accessed.  It never updates a camera, pointmap or state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from versions.v14.b0_person_triangulation import CORE5, closest_rays  # noqa: E402
from versions.v14.b0_person_triangulation_orientation_kabsch import kabsch_rotation  # noqa: E402


DEFAULT_P1 = REPO_ROOT / "output/v14/fine_alignment_research/p1_foot_scene_observability_v2"
DEFAULT_P2 = REPO_ROOT / "output/v14/fine_alignment_research/p2_native_token_who"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/p3_ray_rigid_person_observability"
Q25 = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-cache-dir", type=Path, default=DEFAULT_P1)
    parser.add_argument("--p2-cache-dir", type=Path, default=DEFAULT_P2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ensure_workspace(path: Path) -> None:
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"P3 artifacts must stay in Movie3R workspace: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def transform_points(points: Any, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def copy_geometry(person: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(person[key], dtype=np.float64).copy() for key in ("root", "joints", "vertices")}


def ray_targets(
    pre_person: dict[str, Any], post_person: dict[str, Any], pre_camera: np.ndarray, post_camera: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return all-five CORE5 closest-point targets or an explicit runtime fallback."""
    pre_camera, post_camera = np.asarray(pre_camera, dtype=np.float64), np.asarray(post_camera, dtype=np.float64)
    pre_joints, post_joints = np.asarray(pre_person["joints"], dtype=np.float64), np.asarray(post_person["joints"], dtype=np.float64)
    targets, gaps, sines, depths = [], [], [], []
    for joint_id in CORE5:
        if joint_id >= len(pre_joints) or joint_id >= len(post_joints):
            return None, {"reason": "missing_core_joint", "valid_count": len(targets)}
        direction_pre = pre_joints[joint_id] - pre_camera[:3, 3]
        direction_post = post_joints[joint_id] - post_camera[:3, 3]
        if min(float(np.linalg.norm(direction_pre)), float(np.linalg.norm(direction_post))) <= 1e-8:
            return None, {"reason": "zero_joint_ray", "valid_count": len(targets)}
        midpoint, depth_pre, depth_post, gap, sine = closest_rays(
            pre_camera[:3, 3], direction_pre, post_camera[:3, 3], direction_post
        )
        if depth_pre <= 0.0 or depth_post <= 0.0 or sine <= 1e-5 or not np.isfinite(midpoint).all():
            return None, {"reason": "invalid_ray_intersection", "valid_count": len(targets)}
        targets.append(midpoint)
        gaps.append(float(gap)); sines.append(float(sine)); depths.append((float(depth_pre), float(depth_post)))
    return np.asarray(targets, dtype=np.float64), {
        "reason": "ok", "valid_count": len(targets), "median_gap_m": float(np.median(gaps)),
        "max_gap_m": float(np.max(gaps)), "median_sine": float(np.median(sines)), "min_sine": float(np.min(sines)),
        "depths": depths,
    }


def apply_full_rigid(person: dict[str, Any], source_core: np.ndarray, target_core: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rotation = kabsch_rotation(source_core - source_core.mean(axis=0), target_core - target_core.mean(axis=0))
    translation = target_core.mean(axis=0) - source_core.mean(axis=0) @ rotation.T
    return {
        key: transform_points(person[key], rotation, translation) for key in ("root", "joints", "vertices")
    }, {"rotation_world": rotation, "translation_world": translation, "raw_angle_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))}


def apply_q25_orientation(person: dict[str, Any], source_core: np.ndarray, target_core: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    full_rotation = kabsch_rotation(source_core - source_core.mean(axis=0), target_core - target_core.mean(axis=0))
    raw_vector = Rotation.from_matrix(full_rotation).as_rotvec()
    rotation = Rotation.from_rotvec(raw_vector * Q25).as_matrix()
    output = copy_geometry(person)
    root = output["root"].copy()
    output["joints"] = (output["joints"] - root) @ rotation.T + root
    output["vertices"] = (output["vertices"] - root) @ rotation.T + root
    return output, {"rotation_world": rotation, "raw_angle_deg": float(np.degrees(np.linalg.norm(raw_vector))), "applied_angle_deg": float(np.degrees(np.linalg.norm(raw_vector) * Q25))}


def point_errors(person: dict[str, np.ndarray], target: dict[str, Any]) -> dict[str, float]:
    root = float(np.linalg.norm(person["root"] - np.asarray(target["root_world"], dtype=np.float64)))
    joints_target, vertices_target = np.asarray(target["joints_world"], dtype=np.float64), np.asarray(target["vertices_world"], dtype=np.float64)
    joint_count, vertex_count = min(len(person["joints"]), len(joints_target)), min(len(person["vertices"]), len(vertices_target))
    return {
        "root_m": root,
        "joint_m": float(np.linalg.norm(person["joints"][:joint_count] - joints_target[:joint_count], axis=1).mean()),
        "vertex_m": float(np.linalg.norm(person["vertices"][:vertex_count] - vertices_target[:vertex_count], axis=1).mean()),
    }


def mean_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, float | int]:
    values = [row["metrics_evaluator_only"][method] for row in rows if row.get("has_evaluator_target")]
    if not values:
        return {"count": 0, "root_m": float("nan"), "joint_m": float("nan"), "vertex_m": float("nan")}
    return {"count": len(values), **{key: float(np.mean([value[key] for value in values])) for key in ("root_m", "joint_m", "vertex_m")}}


def harm_rate(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    deltas = []
    for row in rows:
        metrics = row.get("metrics_evaluator_only")
        if metrics is not None:
            deltas.append(float(metrics[method][metric] - metrics["brtc"][metric]))
    return float(np.mean(np.asarray(deltas) > .05)) if deltas else float("nan")


def load_p2_labels(path: Path, expected_hash: str) -> tuple[dict[int, str], dict[int, str]]:
    cached = torch.load(path, map_location="cpu", weights_only=False)
    runtime = cached["runtime"]
    if runtime["b0_camera_sha256"] != expected_hash:
        raise RuntimeError(f"P1/P2 B0 mismatch for {path.name}")
    evaluator = cached["evaluator"]
    return evaluator["pre_labels_by_detection"], evaluator["post_labels_by_detection"]


def build_report(args: argparse.Namespace) -> Path:
    p1_index_path, p2_index_path = args.p1_cache_dir / "P1_CACHE_INDEX.json", args.p2_cache_dir / "P2_CACHE_INDEX.json"
    p1_index, p2_index = json.loads(p1_index_path.read_text(encoding="utf-8")), json.loads(p2_index_path.read_text(encoding="utf-8"))
    if int(p1_index.get("schema", -1)) != 2:
        raise RuntimeError("P3 requires the corrected P1 schema-2 cache")
    p2_paths = {Path(path).stem: Path(path) for path in p2_index["case_paths"]}
    rows, camera_hashes, fallback = [], {}, {}
    for p1_path_text in p1_index["case_paths"]:
        cached = torch.load(p1_path_text, map_location="cpu", weights_only=False)
        if cached.get("status") != "ok":
            continue
        runtime = cached["runtime"]
        if runtime["runtime_contract"]["gt_used"] or runtime["runtime_contract"]["future_post_frames_used"]:
            raise RuntimeError("invalid P1 runtime contract")
        event_id = str(runtime["record"]["event_id"])
        expected_hash = str(runtime["b0_camera_sha256"])
        if array_sha256(runtime["b0_camera_c2w"]) != expected_hash:
            raise RuntimeError(f"camera mutation before P3 action: {event_id}")
        camera_hashes[event_id] = expected_hash
        accepted_by_post = {int(item["post_index"]): bool(item["accepted"]) for item in runtime["brtc"]["people"]}
        # Runtime transaction: no evaluator payload is touched until every action below exists.
        runtime_rows = []
        for pre_index, post_index in runtime["association"]["pairs"]:
            pre_index, post_index = int(pre_index), int(post_index)
            pre_person, b0_person, brtc_person = runtime["pre_people"][pre_index], runtime["b0_post_people"][post_index], runtime["brtc_post_people"][post_index]
            targets, ray_debug = ray_targets(pre_person, b0_person, runtime["pre_camera_c2w"], runtime["b0_camera_c2w"])
            accepted = bool(accepted_by_post.get(post_index, False))
            full, q25, full_debug, q25_debug = copy_geometry(b0_person), copy_geometry(brtc_person), None, None
            reason = str(ray_debug["reason"])
            if accepted and targets is not None:
                source = np.asarray(b0_person["joints"], dtype=np.float64)[list(CORE5)]
                full, full_debug = apply_full_rigid(b0_person, source, targets)
                q25, q25_debug = apply_q25_orientation(brtc_person, source, targets)
                reason = "applied"
            else:
                reason = "brtc_rejected_exact_b0_fallback" if not accepted else reason
            fallback[reason] = fallback.get(reason, 0) + 1
            runtime_rows.append({
                "event_id": event_id, "pre_index": pre_index, "post_index": post_index,
                "pre_detection_index": int(pre_person["detection_index"]), "post_detection_index": int(b0_person["detection_index"]),
                "brtc_accepted": accepted, "ray": ray_debug, "runtime_reason": reason,
                "b0": copy_geometry(b0_person), "brtc": copy_geometry(brtc_person), "ray_rigid_se3_full": full,
                "brtc_ray_target_so3_q25": q25, "full_debug": full_debug, "q25_debug": q25_debug,
            })
        if array_sha256(runtime["b0_camera_c2w"]) != expected_hash:
            raise RuntimeError(f"camera mutation after P3 action: {event_id}")
        # Evaluator-only branch starts here; it cannot influence already-created actions.
        if event_id not in p2_paths:
            raise RuntimeError(f"missing matching P2 case for {event_id}")
        pre_labels, post_labels = load_p2_labels(p2_paths[event_id], expected_hash)
        targets_by_detection = cached["evaluator"]["target_by_detection"]
        for row in runtime_rows:
            target = targets_by_detection.get(int(row["post_detection_index"]))
            pre_id, post_id = pre_labels.get(int(row["pre_detection_index"])), post_labels.get(int(row["post_detection_index"]))
            row["geometry_match_correct_evaluator_only"] = bool(pre_id is not None and post_id is not None and pre_id == post_id)
            row["has_evaluator_target"] = target is not None
            if target is not None:
                row["metrics_evaluator_only"] = {name: point_errors(row[name], target) for name in ("b0", "brtc", "ray_rigid_se3_full", "brtc_ray_target_so3_q25")}
            for key in ("b0", "brtc", "ray_rigid_se3_full", "brtc_ray_target_so3_q25"):
                row.pop(key, None)
            rows.append(row)
    all_rows = [row for row in rows if row.get("has_evaluator_target")]
    correct_rows = [row for row in all_rows if row["geometry_match_correct_evaluator_only"] and row["brtc_accepted"] and row["runtime_reason"] == "applied"]
    methods = ("b0", "brtc", "ray_rigid_se3_full", "brtc_ray_target_so3_q25")
    summaries = {
        "all_geometry_pairs": {name: mean_metrics(all_rows, name) for name in methods},
        "correct_geometry_matches_five_ray_brtc_accepted": {name: mean_metrics(correct_rows, name) for name in methods},
    }
    full, q25, brtc = (summaries["correct_geometry_matches_five_ray_brtc_accepted"][name] for name in ("ray_rigid_se3_full", "brtc_ray_target_so3_q25", "brtc"))
    gate = {
        "correct_five_ray_brtc_accepted_at_least_24": len(correct_rows) >= 24,
        "full_root_improves_brtc_by_5mm": bool(full["root_m"] <= brtc["root_m"] - .005),
        "q25_joint_improves_brtc_by_5mm": bool(q25["joint_m"] <= brtc["joint_m"] - .005),
        "q25_vertex_improves_brtc_by_5mm": bool(q25["vertex_m"] <= brtc["vertex_m"] - .005),
        "full_no_metric_harm_over_5cm": all(harm_rate(correct_rows, "ray_rigid_se3_full", metric) == 0.0 for metric in ("root_m", "joint_m", "vertex_m")),
        "q25_no_metric_harm_over_5cm": all(harm_rate(correct_rows, "brtc_ray_target_so3_q25", metric) == 0.0 for metric in ("root_m", "joint_m", "vertex_m")),
        "camera_bit_exact": True,
        "runtime_actions_before_evaluator": True,
    }
    report = {
        "experiment": "v14_p3_ray_rigid_person_observability", "status": "GO_TO_PAIR_DISJOINT_POLICY" if all(gate.values()) else "NO_GO_RAY_RIGID_PERSON_OBSERVABILITY",
        "p1_cache_index": str(p1_index_path), "p1_cache_index_sha256": sha256(p1_index_path),
        "p2_cache_index": str(p2_index_path), "p2_cache_index_sha256": sha256(p2_index_path),
        "policy": {"core_joints": list(CORE5), "full_action": "raw B0 core5 -> ray midpoint SE3", "orientation_action": "BRTC root + 0.25 ray-target SO3", "fraction": Q25, "selection": "none; fixed observability candidates"},
        "counts": {"rows_with_target": len(all_rows), "correct_five_ray_brtc_accepted": len(correct_rows), "fallback_reasons": fallback},
        "summaries": summaries,
        "harm_vs_brtc_evaluator_only": {name: {metric: harm_rate(correct_rows, name, metric) for metric in ("root_m", "joint_m", "vertex_m")} for name in ("ray_rigid_se3_full", "brtc_ray_target_so3_q25")},
        "runtime_invariants": {"camera_sha256_by_event": camera_hashes, "camera_max_abs_change": 0.0, "all_actions_before_gt": True, "future_post_frames_used": 0, "external_pretrained_models": []},
        "gate": gate, "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "P3_RAY_RIGID_PERSON_REPORT.json"
    destination.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    for path in (args.p1_cache_dir, args.p2_cache_dir, args.output_dir):
        ensure_workspace(path)
    print(build_report(args), flush=True)


if __name__ == "__main__":
    main()
