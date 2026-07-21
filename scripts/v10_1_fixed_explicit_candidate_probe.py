#!/usr/bin/env python3
"""V10.1 probe for a fixed library of explicit cross-shot SE(3) candidates.

Human3R is kept frozen and is reset at the known AABB boundary.  Each explicit
candidate estimates one transform for the complete post-cut shot.  Ground-truth
camera poses are used only after candidate generation to measure errors and to
perform oracle selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.smpl_model import SMPLModel  # noqa: E402
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    AlignmentCandidate,
    average_rotations,
    background_cloud,
    history_background_cloud,
    robust_local_pointmap_refinement,
    root_pose_world,
    se3_magnitude,
)
from v10_token_alignment_4source_probe import (  # noqa: E402
    load_aabb_views_for_record,
)
from v9_learned_stream_alignment_4source_probe import (  # noqa: E402
    aabb_tuple_from_any_record,
    bad_sample_key,
    load_bad_sample_keys,
    run_local_reset_human3r,
    safe_name,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view, output_complete  # noqa: E402
from v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402
from v10_oracle_state_vs_gauge_probe import load_pose, rotation_error_deg  # noqa: E402


DEFAULT_MANIFEST_MAP = (
    REPO_ROOT / "config" / "manifests" / "v10_oracle_candidate_selection_gt_sources" / "manifest_map.json"
)
DEFAULT_BAD_SAMPLE_REGISTRY = REPO_ROOT / "config" / "manifests" / "v10_static_alignment_bad_samples.jsonl"
ANGLE_BUCKETS = ("060_090", "090_120", "120_150", "150_180")
FIXED_EXPLICIT_NAME = "human_mean_pointmap_history_standard"


@dataclass
class CandidateResult:
    name: str
    transform: np.ndarray
    diagnostics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_map", type=Path, default=DEFAULT_MANIFEST_MAP)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source",
    )
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--samples_per_bucket", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--cloud_points_per_frame", type=int, default=5000)
    parser.add_argument("--device", default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bad_sample_registry", type=Path, default=DEFAULT_BAD_SAMPLE_REGISTRY)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest_map(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = data.get("source_manifests", data)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"No source_manifests mapping in {path}")
    return {str(source): Path(value) for source, value in mapping.items()}


def stratified_records(args: argparse.Namespace) -> list[dict]:
    manifest_map = load_manifest_map(args.manifest_map)
    sources = list(manifest_map) if not args.sources else list(args.sources)
    missing = sorted(set(sources) - set(manifest_map))
    if missing:
        raise KeyError(f"Sources missing from manifest map: {missing}")
    bad_keys = load_bad_sample_keys(args.bad_sample_registry)
    selected = []
    rng = np.random.default_rng(int(args.seed))
    for source in sources:
        records = []
        for manifest_index, raw in enumerate(read_jsonl(manifest_map[source])):
            record = dict(raw)
            record["source"] = source
            record["source_manifest_index"] = manifest_index
            if bad_sample_key(record) in bad_keys:
                continue
            records.append(record)
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            bucket = str(record.get("angle_bucket", "unknown"))
            by_bucket[bucket].append(record)
        for bucket in ANGLE_BUCKETS:
            pool = by_bucket.get(bucket, [])
            if not pool:
                continue
            order = rng.permutation(len(pool)).tolist()
            take = min(int(args.samples_per_bucket), len(order))
            for local_index in order[:take]:
                record = dict(pool[local_index])
                seq_a, seq_b, start = aabb_tuple_from_any_record(record)
                record.setdefault(
                    "pattern_id",
                    f"{source}_{bucket}_{record.get('group', 'group')}_{start}_{seq_a.split('/')[-1]}_{seq_b.split('/')[-1]}",
                )
                selected.append(record)
    selected.sort(key=lambda row: (str(row["source"]), str(row.get("angle_bucket", "")), str(row["pattern_id"])))
    return selected


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    return model


def build_smpl_model(model: ARCroco3DStereo, device: torch.device) -> SMPLModel:
    return SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )


def predicted_poses(local_dir: Path, frame_count: int = 4) -> np.ndarray:
    return np.stack([load_pose(local_dir, idx)[0] for idx in range(frame_count)]).astype(np.float32)


def target_poses_in_prediction_gauge(views: list[dict], pred_poses: np.ndarray) -> tuple[np.ndarray, dict]:
    gt = np.stack(
        [gt_pose_from_view(view).detach().cpu().numpy().astype(np.float32) for view in views],
        axis=0,
    )
    gauge = pred_poses[0] @ np.linalg.inv(gt[0])
    target = np.stack([(gauge @ pose).astype(np.float32) for pose in gt], axis=0)
    return target, {
        "definition": "GT camera trajectory aligned once to Human3R frame-0 gauge",
        "gauge_transform": gauge.tolist(),
    }


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def human_initial(
    local_dir: Path,
    history_indices: list[int],
    current_idx: int,
    mode: str,
    rotation: bool = True,
    translation: bool = True,
) -> AlignmentCandidate:
    history = [root_pose_world(local_dir, idx) for idx in history_indices]
    current_R, current_t = root_pose_world(local_dir, current_idx)
    if mode == "last":
        target_R, target_t = history[-1]
    elif mode == "mean":
        target_R = average_rotations([item[0] for item in history])
        target_t = np.median(np.stack([item[1] for item in history]), axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported human initial mode: {mode}")
    align_R = target_R @ current_R.T if rotation else np.eye(3, dtype=np.float32)
    align_t = target_t - align_R @ current_t if translation else np.zeros(3, dtype=np.float32)
    name_bits = ["human", mode]
    name_bits.append("full" if rotation and translation else "translation_only" if translation else "rotation_only")
    transform = make_transform(align_R, align_t)
    return AlignmentCandidate(
        name="_".join(name_bits),
        transform=transform,
        confidence=1.0,
        diagnostics={
            "history_indices": history_indices,
            "current_idx": current_idx,
            "target_root": target_t.tolist(),
            "current_root": current_t.tolist(),
            "transform_magnitude": se3_magnitude(transform),
        },
    )


def combine_clouds(local_dir: Path, indices: list[int], max_points: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    clouds = []
    diagnostics = []
    for offset, idx in enumerate(indices):
        cloud, debug = background_cloud(local_dir, idx, max_points, seed + offset)
        clouds.append(cloud)
        diagnostics.append({"idx": idx, **debug})
    valid = [cloud for cloud in clouds if len(cloud)]
    if not valid:
        return np.empty((0, 3), dtype=np.float32), diagnostics
    return np.concatenate(valid, axis=0).astype(np.float32), diagnostics


def refinement_config(name: str) -> SimpleNamespace:
    configs = {
        "standard": (8, 0.60, 0.12),
        "strict": (6, 0.35, 0.08),
        "loose": (10, 1.00, 0.20),
    }
    iterations, maximum, minimum = configs[name]
    return SimpleNamespace(refine_iters=iterations, refine_max_distance=maximum, refine_min_distance=minimum)


def refine_candidate(
    name: str,
    initial: AlignmentCandidate,
    source: np.ndarray,
    target: np.ndarray,
    config_name: str,
    cloud_debug: dict,
) -> CandidateResult:
    if len(source) < 32 or len(target) < 32:
        return CandidateResult(
            name=name,
            transform=initial.transform.copy(),
            diagnostics={
                "status": "too_few_background_points",
                "initial": initial.name,
                "source_points": int(len(source)),
                "target_points": int(len(target)),
                **cloud_debug,
            },
        )
    transform, debug = robust_local_pointmap_refinement(
        initial.transform,
        source,
        target,
        refinement_config(config_name),
    )
    return CandidateResult(
        name=name,
        transform=transform,
        diagnostics={
            "status": "ok",
            "initial": initial.name,
            "config": config_name,
            "source_points": int(len(source)),
            "target_points": int(len(target)),
            "clouds": cloud_debug,
            "refinement": debug,
        },
    )


def build_candidates(local_dir: Path, args: argparse.Namespace, case_seed: int) -> list[CandidateResult]:
    boundary = int(args.boundary)
    history = list(range(boundary))
    last_history = [boundary - 1]
    point_limit = int(args.cloud_points_per_frame)

    human_last = human_initial(local_dir, history, boundary, mode="last")
    human_mean = human_initial(local_dir, history, boundary, mode="mean")
    human_last_t = human_initial(local_dir, history, boundary, mode="last", rotation=False)
    human_mean_t = human_initial(local_dir, history, boundary, mode="mean", rotation=False)
    identity = AlignmentCandidate("identity", np.eye(4, dtype=np.float32), 0.0, {})

    target_last, target_last_debug = history_background_cloud(local_dir, last_history, point_limit)
    target_history, target_history_debug = history_background_cloud(local_dir, history, point_limit)
    source_b1, source_b1_debug = combine_clouds(local_dir, [boundary], point_limit, case_seed + 100)
    source_b12, source_b12_debug = combine_clouds(local_dir, [boundary, boundary + 1], point_limit, case_seed + 200)

    candidates = [
        CandidateResult("identity_fallback", identity.transform.copy(), {"status": "ok"}),
        CandidateResult("human_last_full_no_refine", human_last.transform.copy(), human_last.diagnostics),
        CandidateResult("human_mean_full_no_refine", human_mean.transform.copy(), human_mean.diagnostics),
        CandidateResult("human_last_translation_only", human_last_t.transform.copy(), human_last_t.diagnostics),
        CandidateResult("human_mean_translation_only", human_mean_t.transform.copy(), human_mean_t.diagnostics),
    ]
    candidates.extend(
        [
            refine_candidate(
                "pointmap_identity_last_standard",
                identity,
                source_b1,
                target_last,
                "standard",
                {"source": source_b1_debug, "target": target_last_debug},
            ),
            refine_candidate(
                "human_last_pointmap_last_standard",
                human_last,
                source_b1,
                target_last,
                "standard",
                {"source": source_b1_debug, "target": target_last_debug},
            ),
            refine_candidate(
                FIXED_EXPLICIT_NAME,
                human_mean,
                source_b1,
                target_history,
                "standard",
                {"source": source_b1_debug, "target": target_history_debug},
            ),
            refine_candidate(
                "human_mean_pointmap_history_strict",
                human_mean,
                source_b1,
                target_history,
                "strict",
                {"source": source_b1_debug, "target": target_history_debug},
            ),
            refine_candidate(
                "human_mean_pointmap_history_loose",
                human_mean,
                source_b1,
                target_history,
                "loose",
                {"source": source_b1_debug, "target": target_history_debug},
            ),
            refine_candidate(
                "human_mean_pointmap_wait_b2_standard",
                human_mean,
                source_b12,
                target_history,
                "standard",
                {"source": source_b12_debug, "target": target_history_debug, "fixed_delay_frames": 1},
            ),
            refine_candidate(
                "human_last_pointmap_wait_b2_standard",
                human_last,
                source_b12,
                target_last,
                "standard",
                {"source": source_b12_debug, "target": target_last_debug, "fixed_delay_frames": 1},
            ),
        ]
    )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate candidate names: {names}")
    return candidates


def transform_camera_poses(poses: np.ndarray, transform: np.ndarray, boundary: int) -> np.ndarray:
    output = poses.copy()
    output[boundary:] = np.einsum("ij,njk->nik", transform, poses[boundary:]).astype(np.float32)
    return output


def camera_errors(poses: np.ndarray, target: np.ndarray, boundary: int) -> dict:
    per_frame = []
    for idx in range(boundary, len(poses)):
        per_frame.append(
            {
                "idx": idx,
                "translation_m": float(np.linalg.norm(poses[idx, :3, 3] - target[idx, :3, 3])),
                "rotation_deg": rotation_error_deg(poses[idx], target[idx]),
            }
        )
    mean_t = float(np.mean([row["translation_m"] for row in per_frame]))
    mean_r = float(np.mean([row["rotation_deg"] for row in per_frame]))
    return {
        "mean_translation_m": mean_t,
        "mean_rotation_deg": mean_r,
        "boundary_translation_m": per_frame[0]["translation_m"],
        "boundary_rotation_deg": per_frame[0]["rotation_deg"],
        "joint_oracle_cost": mean_t + math.radians(mean_r),
        "success_strict": bool(mean_t < 0.10 and mean_r < 5.0),
        "success_relaxed": bool(mean_t < 0.25 and mean_r < 10.0),
        "catastrophic": bool(mean_t > 1.0 or mean_r > 30.0),
        "per_frame": per_frame,
    }


def evaluate_candidates(
    candidates: list[CandidateResult],
    pred_poses: np.ndarray,
    target_poses: np.ndarray,
    boundary: int,
) -> tuple[list[dict], dict]:
    rows = []
    oracle_transform = target_poses[boundary] @ np.linalg.inv(pred_poses[boundary])
    for candidate in candidates:
        aligned = transform_camera_poses(pred_poses, candidate.transform, boundary)
        rows.append(
            {
                "name": candidate.name,
                "transform": candidate.transform.tolist(),
                "transform_magnitude": se3_magnitude(candidate.transform),
                "metrics": camera_errors(aligned, target_poses, boundary),
                "diagnostics": candidate.diagnostics,
            }
        )
    rows.sort(key=lambda row: row["metrics"]["joint_oracle_cost"])
    oracle_poses = transform_camera_poses(pred_poses, oracle_transform.astype(np.float32), boundary)
    return rows, {
        "name": "oracle_boundary_se3",
        "transform": oracle_transform.astype(np.float32).tolist(),
        "metrics": camera_errors(oracle_poses, target_poses, boundary),
    }


def case_name(record: dict) -> str:
    return safe_name(str(record["pattern_id"]))


def run_case(
    model: ARCroco3DStereo | None,
    smpl_model: SMPLModel | None,
    record: dict,
    args: argparse.Namespace,
    device: torch.device,
    case_index: int,
) -> dict:
    output_dir = args.output_dir / "cases" / case_name(record)
    metrics_path = output_dir / "case_metrics.json"
    if metrics_path.is_file() and not args.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / "human3r_local_reset"
    views = load_aabb_views_for_record(record, args, device)
    if smpl_model is not None:
        smpl_model.update_smpl_gt(views)
    start = time.perf_counter()
    if args.skip_inference:
        if not output_complete(local_dir):
            raise FileNotFoundError(f"Missing cached Human3R output: {local_dir}")
    else:
        if model is None:
            raise RuntimeError("Model is required unless --skip_inference is used")
        run_local_reset_human3r(model, views, local_dir, args, device)
    inference_seconds = time.perf_counter() - start

    pred = predicted_poses(local_dir)
    target, gauge_debug = target_poses_in_prediction_gauge(views, pred)
    candidates = build_candidates(local_dir, args, int(args.seed) + case_index * 1000)
    evaluated, oracle_se3 = evaluate_candidates(candidates, pred, target, int(args.boundary))
    lookup = {row["name"]: row for row in evaluated}
    fixed = lookup[FIXED_EXPLICIT_NAME]
    oracle_selected = evaluated[0]
    report = {
        "case_name": case_name(record),
        "record": record,
        "candidate_count": len(evaluated),
        "fixed_explicit_name": FIXED_EXPLICIT_NAME,
        "fixed_explicit": fixed,
        "oracle_selected": oracle_selected,
        "oracle_boundary_se3": oracle_se3,
        "oracle_gain": {
            "translation_m": fixed["metrics"]["mean_translation_m"]
            - oracle_selected["metrics"]["mean_translation_m"],
            "rotation_deg": fixed["metrics"]["mean_rotation_deg"]
            - oracle_selected["metrics"]["mean_rotation_deg"],
            "joint_cost": fixed["metrics"]["joint_oracle_cost"]
            - oracle_selected["metrics"]["joint_oracle_cost"],
            "selected_different_candidate": oracle_selected["name"] != FIXED_EXPLICIT_NAME,
        },
        "candidates": evaluated,
        "gt_gauge": gauge_debug,
        "timing": {"human3r_or_cache_seconds": inference_seconds},
        "paths": {"human3r_local_reset": str(local_dir)},
    }
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def aggregate_method(cases: list[dict], key: str) -> dict:
    metrics = [case[key]["metrics"] for case in cases]
    translation = [item["mean_translation_m"] for item in metrics]
    rotation = [item["mean_rotation_deg"] for item in metrics]
    return {
        "count": len(metrics),
        "translation_mean_m": float(np.mean(translation)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": percentile(translation, 90),
        "translation_p95_m": percentile(translation, 95),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": percentile(rotation, 90),
        "rotation_p95_deg": percentile(rotation, 95),
        "success_strict_rate": float(np.mean([item["success_strict"] for item in metrics])),
        "success_relaxed_rate": float(np.mean([item["success_relaxed"] for item in metrics])),
        "catastrophic_rate": float(np.mean([item["catastrophic"] for item in metrics])),
    }


def candidate_for_case(case: dict, name: str) -> dict:
    return next(candidate for candidate in case["candidates"] if candidate["name"] == name)


def enrich_oracle_selections(cases: list[dict]) -> str:
    candidate_names = sorted({candidate["name"] for case in cases for candidate in case["candidates"]})
    mean_joint_cost = {
        name: float(
            np.mean([candidate_for_case(case, name)["metrics"]["joint_oracle_cost"] for case in cases])
        )
        for name in candidate_names
    }
    best_single_name = min(candidate_names, key=lambda name: mean_joint_cost[name])
    for case in cases:
        case["best_single_fixed"] = candidate_for_case(case, best_single_name)
        case["oracle_translation_selected"] = min(
            case["candidates"], key=lambda row: row["metrics"]["mean_translation_m"]
        )
        case["oracle_rotation_selected"] = min(
            case["candidates"], key=lambda row: row["metrics"]["mean_rotation_deg"]
        )
    return best_single_name


def aggregate_group(cases: list[dict]) -> dict:
    winner_counts = Counter(case["oracle_selected"]["name"] for case in cases)
    gains_t = [case["oracle_gain"]["translation_m"] for case in cases]
    gains_r = [case["oracle_gain"]["rotation_deg"] for case in cases]
    gains_joint = [case["oracle_gain"]["joint_cost"] for case in cases]
    result = {
        "count": len(cases),
        "fixed_explicit": aggregate_method(cases, "fixed_explicit"),
        "best_single_fixed": aggregate_method(cases, "best_single_fixed"),
        "oracle_candidate_selection": aggregate_method(cases, "oracle_selected"),
        "oracle_translation_selection": aggregate_method(cases, "oracle_translation_selected"),
        "oracle_rotation_selection": aggregate_method(cases, "oracle_rotation_selected"),
        "oracle_boundary_se3": aggregate_method(cases, "oracle_boundary_se3"),
        "winner_counts": dict(winner_counts.most_common()),
        "different_from_fixed_rate": float(
            np.mean([case["oracle_gain"]["selected_different_candidate"] for case in cases])
        ),
        "mean_translation_gain_m": float(np.mean(gains_t)),
        "mean_rotation_gain_deg": float(np.mean(gains_r)),
        "mean_joint_cost_gain": float(np.mean(gains_joint)),
        "positive_joint_gain_rate": float(np.mean([gain > 1e-6 for gain in gains_joint])),
    }
    best_metrics = [case["best_single_fixed"]["metrics"] for case in cases]
    oracle_metrics = [case["oracle_selected"]["metrics"] for case in cases]
    best_names = [case["best_single_fixed"]["name"] for case in cases]
    result["oracle_vs_best_single"] = {
        "best_single_name": best_names[0],
        "different_candidate_rate": float(
            np.mean([oracle["name"] != best for oracle, best in zip([case["oracle_selected"] for case in cases], best_names)])
        ),
        "mean_translation_gain_m": float(
            np.mean([best["mean_translation_m"] - oracle["mean_translation_m"] for best, oracle in zip(best_metrics, oracle_metrics)])
        ),
        "mean_rotation_gain_deg": float(
            np.mean([best["mean_rotation_deg"] - oracle["mean_rotation_deg"] for best, oracle in zip(best_metrics, oracle_metrics)])
        ),
        "mean_joint_cost_gain": float(
            np.mean([best["joint_oracle_cost"] - oracle["joint_oracle_cost"] for best, oracle in zip(best_metrics, oracle_metrics)])
        ),
    }
    return result


def aggregate_report(cases: list[dict], args: argparse.Namespace) -> dict:
    best_single_name = enrich_oracle_selections(cases)
    by_source = {}
    for source in sorted({str(case["record"]["source"]) for case in cases}):
        subset = [case for case in cases if str(case["record"]["source"]) == source]
        by_source[source] = aggregate_group(subset)
    by_angle = {}
    for bucket in ANGLE_BUCKETS:
        subset = [case for case in cases if str(case["record"].get("angle_bucket")) == bucket]
        if subset:
            by_angle[bucket] = aggregate_group(subset)
    return {
        "experiment": "V10.1 Fixed Explicit Candidate Probe",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "candidate_policy": {
            "fixed_explicit": FIXED_EXPLICIT_NAME,
            "best_single_fixed_selected_posthoc": best_single_name,
            "oracle_objective": "mean translation error in meters + mean rotation error in radians",
            "gt_used_for_candidate_generation": False,
            "shot_level_transform_only": True,
            "post_cut_state": "fresh Human3R state",
        },
        "overall": aggregate_group(cases),
        "by_source": by_source,
        "by_angle_bucket": by_angle,
        "cases": cases,
    }


def write_flat_csv(path: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        fixed = case["fixed_explicit"]["metrics"]
        best = case["best_single_fixed"]["metrics"]
        oracle = case["oracle_selected"]["metrics"]
        upper = case["oracle_boundary_se3"]["metrics"]
        rows.append(
            {
                "case_name": case["case_name"],
                "source": case["record"]["source"],
                "group": case["record"].get("group"),
                "angle_bucket": case["record"].get("angle_bucket"),
                "view_angle_deg": case["record"].get("view_angle_deg"),
                "fixed_t_m": fixed["mean_translation_m"],
                "fixed_r_deg": fixed["mean_rotation_deg"],
                "best_single_candidate": case["best_single_fixed"]["name"],
                "best_single_t_m": best["mean_translation_m"],
                "best_single_r_deg": best["mean_rotation_deg"],
                "oracle_candidate": case["oracle_selected"]["name"],
                "oracle_t_m": oracle["mean_translation_m"],
                "oracle_r_deg": oracle["mean_rotation_deg"],
                "oracle_boundary_t_m": upper["mean_translation_m"],
                "oracle_boundary_r_deg": upper["mean_rotation_deg"],
                "gain_t_m": case["oracle_gain"]["translation_m"],
                "gain_r_deg": case["oracle_gain"]["rotation_deg"],
                "gain_joint": case["oracle_gain"]["joint_cost"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_line(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['translation_mean_m']:.4f} | {metrics['translation_median_m']:.4f} | "
        f"{metrics['translation_p90_m']:.4f} | {metrics['rotation_mean_deg']:.2f} | "
        f"{metrics['rotation_median_deg']:.2f} | {metrics['rotation_p90_deg']:.2f} | "
        f"{100.0 * metrics['success_relaxed_rate']:.1f}% | {100.0 * metrics['catastrophic_rate']:.1f}% |"
    )


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    lines = [
        "# V10.1 Fixed Explicit Candidate Probe",
        "",
        "## 设置",
        "",
        "- Human3R 主体冻结，GT boundary 固定在 AABB 的第 3 帧前。",
        "- cut 后使用 fresh recurrent state。每个候选只估计一次 SE(3)，并固定作用于整个 B 段。",
        "- GT camera 只用于评测和 Oracle 选择，不参与候选生成。",
        f"- 固定 Explicit baseline：`{FIXED_EXPLICIT_NAME}`。",
        "- Oracle 目标：`平均平移误差(m) + 平均旋转误差(rad)`。",
        "",
        "## 总体结果",
        "",
        "| 方法 | T mean | T median | T P90 | R mean | R median | R P90 | Relaxed success | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        method_line("Fixed Explicit", overall["fixed_explicit"]),
        method_line("Best Single Fixed", overall["best_single_fixed"]),
        method_line("Oracle Candidate Selection", overall["oracle_candidate_selection"]),
        method_line("Oracle Translation Selection", overall["oracle_translation_selection"]),
        method_line("Oracle Rotation Selection", overall["oracle_rotation_selection"]),
        method_line("Oracle boundary SE(3)", overall["oracle_boundary_se3"]),
        "",
        f"- Oracle 与固定候选不同的比例：`{100.0 * overall['different_from_fixed_rate']:.1f}%`。",
        f"- Oracle 平均平移改善：`{overall['mean_translation_gain_m']:.4f} m`。",
        f"- Oracle 平均旋转改善：`{overall['mean_rotation_gain_deg']:.2f}°`。",
        f"- Oracle joint cost 有正改善的比例：`{100.0 * overall['positive_joint_gain_rate']:.1f}%`。",
        f"- 事后最强单一固定候选：`{overall['oracle_vs_best_single']['best_single_name']}`。",
        f"- 相对最强单一固定候选，Oracle joint cost 平均改善：`{overall['oracle_vs_best_single']['mean_joint_cost_gain']:.4f}`。",
        f"- 相对最强单一固定候选，Oracle 平移改善：`{overall['oracle_vs_best_single']['mean_translation_gain_m']:.4f} m`，旋转改善：`{overall['oracle_vs_best_single']['mean_rotation_gain_deg']:.2f}°`。",
        f"- Oracle 与最强单一固定候选不同的比例：`{100.0 * overall['oracle_vs_best_single']['different_candidate_rate']:.1f}%`。",
        "",
        "## Oracle 候选胜出次数",
        "",
    ]
    for name, count in overall["winner_counts"].items():
        lines.append(f"- `{name}`：{count}")
    lines.extend(["", "## 分数据源", ""])
    for source, group in report["by_source"].items():
        fixed = group["fixed_explicit"]
        oracle = group["oracle_candidate_selection"]
        lines.append(
            f"- **{source}** ({group['count']}): Fixed `{fixed['translation_mean_m']:.3f} m / {fixed['rotation_mean_deg']:.1f}°`; "
            f"Oracle `{oracle['translation_mean_m']:.3f} m / {oracle['rotation_mean_deg']:.1f}°`; "
            f"不同候选率 `{100.0 * group['different_from_fixed_rate']:.1f}%`."
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            "候选互补性成立，但当前不应直接训练 Selector。",
            "Oracle 相对最强单一固定候选的上界仍然有限，relaxed success 和 catastrophic failure 基本没有改善，",
            "说明当前主要瓶颈仍是候选本身缺少正确解。下一步先增加人体朝向歧义、root 深度/尺度、地面重力和多帧运动候选。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = stratified_records(args)
    if not records:
        raise RuntimeError("No records selected")
    (args.output_dir / "selected_records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        f">> selected {len(records)} cases; "
        f"sources={sorted({record['source'] for record in records})}; device={args.device}",
        flush=True,
    )
    device = torch.device(args.device)
    model = None if args.skip_inference else build_model(args)
    smpl_model = None if model is None else build_smpl_model(model, device)
    cases = []
    failures = []
    progress_path = args.output_dir / "case_progress.jsonl"
    if args.overwrite and progress_path.exists():
        progress_path.unlink()
    for index, record in enumerate(records):
        print(
            f">> [{index + 1}/{len(records)}] {record['source']} "
            f"{record.get('angle_bucket')} {record['pattern_id']}",
            flush=True,
        )
        try:
            case = run_case(model, smpl_model, record, args, device, index)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            failure = {"record": record, "error": str(exc)}
            failures.append(failure)
            print(f"!! skip case: {exc}", flush=True)
            continue
        cases.append(case)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "case_name": case["case_name"],
                        "source": record["source"],
                        "winner": case["oracle_selected"]["name"],
                        "fixed_t": case["fixed_explicit"]["metrics"]["mean_translation_m"],
                        "fixed_r": case["fixed_explicit"]["metrics"]["mean_rotation_deg"],
                        "oracle_t": case["oracle_selected"]["metrics"]["mean_translation_m"],
                        "oracle_r": case["oracle_selected"]["metrics"]["mean_rotation_deg"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not cases:
        raise RuntimeError(f"All cases failed: {failures[:5]}")
    report = aggregate_report(cases, args)
    report["failures"] = failures
    (args.output_dir / "oracle_candidate_selection_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_flat_csv(args.output_dir / "oracle_candidate_selection_cases.csv", cases)
    write_markdown(args.output_dir / "oracle_candidate_selection_metrics.md", report)
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False), flush=True)
    print(f">> report: {args.output_dir / 'oracle_candidate_selection_metrics.md'}", flush=True)


if __name__ == "__main__":
    main()
