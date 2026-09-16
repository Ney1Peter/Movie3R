#!/usr/bin/env python3
"""Fit R1--R3 from a sealed R0 cache and materialize evaluator inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.registration_baselines.geometry import (  # noqa: E402
    RegistrationResult,
    apply_shared_transform,
    human_joint_se3,
    pelvis_translation,
    rotation_degrees,
    scene_fpfh_icp,
    transform_points,
)


SCHEMA = "Shot3R-traditional-registration-prediction-cache-v1"
CHECKPOINT_SHA256 = "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377"
ARRAY_KEYS = (
    "cameras_c2w", "vertices_world", "joints_world",
    "persistent_ids", "native_ids", "valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--r0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test", "smoke"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--lines")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_r0(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: np.asarray(cache[f"r0__{key}"]) for key in ARRAY_KEYS}
        scene = {
            "pre": np.asarray(cache["scene_pre_points"]),
            "post": np.asarray(cache["scene_post_points"]),
        }
    return arrays, scene


def method_payload(
    dataset: str, case_id: str, method: str, kind: str, access: str,
    record: dict[str, Any], result: RegistrationResult, manifest_sha256: str,
    config: dict[str, Any], arrays: dict[str, np.ndarray], scene: dict[str, np.ndarray],
) -> dict[str, Any]:
    boundary = int(record["boundary_index"])
    rotation = rotation_degrees(result.transform)
    translation = float(np.linalg.norm(result.transform[:3, 3]))
    diagnostics = dict(result.diagnostics)
    return {
        "schema_version": "Shot3R-traditional-registration-transform-v1",
        "dataset": dataset,
        "case_id": case_id,
        "method_id": method,
        "runtime_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "boundary_index": boundary,
        "temporal_access": access,
        "gt_used_for_registration": False,
        "transform_kind": kind,
        "transform_B_to_A": np.asarray(result.transform).tolist(),
        "scale": 1.0,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "fallback": "R0 identity" if result.status != "ok" else None,
        "person_count_pre": int(np.asarray(arrays["valid"])[boundary - 1].sum()),
        "person_count_post": int(np.asarray(arrays["valid"])[boundary].sum()),
        "matched_person_count": int(diagnostics.get("matched_person_count", 0)),
        "scene_point_count_pre": int(diagnostics.get("scene_point_count_pre", len(scene["pre"]))),
        "scene_point_count_post": int(diagnostics.get("scene_point_count_post", len(scene["post"]))),
        "correspondence_count": int(diagnostics.get("correspondence_count", 0)),
        "inlier_ratio": diagnostics.get("inlier_ratio"),
        "fit_residual": diagnostics.get("fit_residual"),
        "runtime_seconds": float(diagnostics.get("runtime_seconds", 0.0)),
        "estimated_translation_m": translation,
        "estimated_rotation_deg": rotation,
        "random_seed": int(config["random_seed"]),
        "configuration_id": str(config["configuration_id"]),
        "diagnostics": diagnostics,
        "id_map": {str(key): int(value) for key, value in result.id_map.items()},
    }


def process_case(
    record: dict[str, Any], r0_path: Path, output_root: Path,
    dataset: str, split: str, config: dict[str, Any], manifest_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_id = str(record["case_id"])
    boundary = int(record["boundary_index"])
    arrays, scene = load_r0(r0_path)
    r0 = RegistrationResult(
        np.eye(4, dtype=np.float64), "ok", None,
        {"runtime_seconds": 0.0}, {},
    )
    r1_cfg, r2_cfg, r3_cfg = config["r1"], config["r2"], config["r3"]
    r1 = pelvis_translation(arrays, boundary, float(r1_cfg["max_pair_cost"]))
    r2 = human_joint_se3(
        arrays, boundary, float(r2_cfg["max_pair_cost"]),
        float(r2_cfg["inlier_threshold_m"]), int(r2_cfg["ransac_trials"]),
        float(r2_cfg["min_inlier_ratio"]), int(config["random_seed"]),
    )
    r3 = scene_fpfh_icp(
        scene["post"], scene["pre"], float(r3_cfg["voxel_size_m"]),
        float(r3_cfg["ransac_distance_factor"]), float(r3_cfg["icp_distance_factor"]),
        int(r3_cfg["min_scene_points"]), int(r3_cfg["min_correspondences"]),
        float(r3_cfg["min_fitness"]), float(r3_cfg["max_inlier_rmse_m"]),
        int(config["random_seed"]),
    )
    results = {"r0": r0, "r1": r1, "r2": r2, "r3": r3}
    specifications = {
        "r0": ("SE3", "per-shot online"),
        "r1": ("translation", "boundary-time"),
        "r2": ("SE3", "boundary-time"),
        "r3": ("SE3", "offline post-hoc"),
    }
    packed: dict[str, np.ndarray] = {}
    transforms = {}
    for method, result in results.items():
        mapped = apply_shared_transform(arrays, result.transform, boundary, result.id_map)
        for key, value in mapped.items():
            packed[f"{method}__{key}"] = value
        # The pre-shot cloud already defines frame A. The same B-to-A transform
        # used for cameras and people is applied to the complete post-shot cloud.
        packed[f"{method}__scene_pre_points"] = np.asarray(scene["pre"])
        packed[f"{method}__scene_post_points"] = transform_points(
            np.asarray(scene["post"]), result.transform
        )
        kind, access = specifications[method]
        payload = method_payload(
            dataset, case_id, method, kind, access, record, result,
            manifest_sha256, config, arrays, scene,
        )
        path = output_root / "transforms" / dataset / split / case_id / f"{method}.json"
        atomic_json(path, payload)
        transforms[method] = payload
    prediction_path = output_root / "predictions" / dataset / split / f"{case_id}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    partial = prediction_path.with_suffix(prediction_path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(partial, prediction_path)
    runtime_path = prediction_path.with_suffix(".runtime.json")
    atomic_json(runtime_path, {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "dataset": dataset,
        "split": split,
        "record": record,
        "methods": list(results),
        "source_r0_cache": str(r0_path.resolve()),
        "prediction_cache": str(prediction_path.resolve()),
        "configuration": config,
        "runtime_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gt_used_for_registration": False,
        "transforms": {
            method: {
                "status": value["status"],
                "failure_reason": value["failure_reason"],
                "runtime_seconds": value["runtime_seconds"],
            }
            for method, value in transforms.items()
        },
        "total_postprocess_seconds": time.perf_counter() - started,
    })
    return {
        "case_id": case_id,
        "prediction": str(prediction_path),
        "runtime": str(runtime_path),
        "statuses": {method: result.status for method, result in results.items()},
        "seconds": time.perf_counter() - started,
    }


def reusable(path: Path, runtime: Path, case_id: str, configuration_id: str) -> bool:
    if not path.is_file() or not runtime.is_file():
        return False
    try:
        report = json.loads(runtime.read_text(encoding="utf-8"))
        if report.get("case_id") != case_id:
            return False
        if report.get("configuration", {}).get("configuration_id") != configuration_id:
            return False
        with np.load(path, allow_pickle=False) as cache:
            return all(
                f"{method}__{key}" in cache.files
                for method in ("r0", "r1", "r2", "r3")
                for key in (*ARRAY_KEYS, "scene_pre_points", "scene_post_points")
            )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    rows = read_rows(args.runtime_manifest.resolve())
    config = json.loads(args.config.read_text(encoding="utf-8"))
    requested = {int(value) for value in args.lines.split(",") if value.strip()} if args.lines else None
    selected = [
        (index, row) for index, row in enumerate(rows, start=1)
        if (requested is None or index in requested)
        and (index - 1) % args.num_shards == args.shard_index
    ]
    if args.max_cases is not None:
        selected = selected[: int(args.max_cases)]
    completed, reused, failures = [], [], []
    for position, (line, record) in enumerate(selected, start=1):
        case_id = str(record["case_id"])
        r0_path = args.r0_root / args.dataset / args.split / f"{case_id}.r0_scene.npz"
        output = args.output_root / "predictions" / args.dataset / args.split / f"{case_id}.npz"
        runtime = output.with_suffix(".runtime.json")
        if reusable(output, runtime, case_id, str(config["configuration_id"])):
            reused.append(case_id)
            print(f"[{position}/{len(selected)}] reusable {case_id}", flush=True)
            continue
        try:
            result = process_case(
                record, r0_path, args.output_root, args.dataset, args.split,
                config, args.manifest_sha256,
            )
            completed.append(result)
            print(f"[{position}/{len(selected)}] complete {case_id} {result['statuses']} {result['seconds']:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"case_id": case_id, "line": line, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{position}/{len(selected)}] FAILED {case_id}: {failures[-1]['error']}", flush=True)
    ledger = args.output_root / "logs" / f"fit_{args.dataset}_{args.split}_shard{args.shard_index:02d}.json"
    atomic_json(ledger, {
        "schema_version": "Shot3R-registration-fit-shard-ledger-v1",
        "dataset": args.dataset, "split": args.split,
        "configuration": config, "selected_count": len(selected),
        "completed": completed, "reused": reused, "failures": failures,
    })
    if failures:
        raise SystemExit(f"{len(failures)} cases failed; see {ledger}")


if __name__ == "__main__":
    main()
