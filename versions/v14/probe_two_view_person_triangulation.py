#!/usr/bin/env python3
"""Causal, camera-frozen two-view person-depth triangulation probe.

The method uses only the last pre-cut frame and the first post-cut frame.  It
casts rays through corresponding predicted SMPL-X core joints using a frozen
camera trajectory, triangulates each joint, transfers the triangulated point
back to a candidate post-cut pelvis, and applies at most one rigid translation
along the post-cut pelvis ray.  It never changes a camera, pose, shape, or the
pre-cut person.

This archive probe has two deliberately separate phases:

* ``--phase dev`` reads offset0 only and freezes a fully observable gate;
* ``--phase confirm`` reads offset50 only and requires the frozen JSON.

GT people are used only by the evaluator.  They never enter ray construction,
triangulation, aggregation, acceptance, or the action magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from smplx.joint_names import JOINT_NAMES

from probe_controlled_token_person_residual import (
    DEFAULT_DATA,
    DEFAULT_HELDOUT_ARCHIVE,
    DEFAULT_RAW_ROOTS,
    DEFAULT_TRAIN_ARCHIVE,
    REPO_ROOT,
    corrected_root_metrics,
    gt_local_joints,
    json_ready,
    load_archive_metadata,
    predicted_joints,
    _load_avatarrex_raw_calibration,
)


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/two_view_person_triangulation"
CORE_JOINT_SETS = {
    "pelvis": (0,),
    "torso5": (0, 1, 2, 16, 17),
    "core11": (0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--train_archive", type=Path, default=DEFAULT_TRAIN_ARCHIVE)
    parser.add_argument("--heldout_archive", type=Path, default=DEFAULT_HELDOUT_ARCHIVE)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smpl_chunk", type=int, default=32)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def closest_rays(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Return midpoint, two signed depths, ray gap, and conditioning sine."""
    direction_a = direction_a / max(float(np.linalg.norm(direction_a)), 1e-12)
    direction_b = direction_b / max(float(np.linalg.norm(direction_b)), 1e-12)
    system = np.stack((direction_a, -direction_b), axis=1)
    depths, _, _, _ = np.linalg.lstsq(system, origin_b - origin_a, rcond=None)
    point_a = origin_a + float(depths[0]) * direction_a
    point_b = origin_b + float(depths[1]) * direction_b
    dot = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - dot * dot)))
    return (
        0.5 * (point_a + point_b),
        float(depths[0]),
        float(depths[1]),
        float(np.linalg.norm(point_a - point_b)),
        sine,
    )


def build_probe_payload(
    name: str,
    archive: Path,
    data_root: Path,
    device: torch.device,
    smpl_chunk: int,
) -> dict[str, Any]:
    raw_calibration = _load_avatarrex_raw_calibration(DEFAULT_RAW_ROOTS)
    records, frames_by_sample, _ = load_archive_metadata(
        name, archive, data_root, raw_calibration
    )
    flat = [frame for sample in frames_by_sample for frame in sample]
    pred = predicted_joints(flat, device, smpl_chunk)
    gt = gt_local_joints(flat, device, smpl_chunk)
    pred = pred.reshape(len(records), 4, pred.shape[1], 3)
    gt = gt.reshape(len(records), 4, gt.shape[1], 3)

    post_root = pred[:, 2, 0]
    gt_post_root = gt[:, 2, 0]
    post_ray = post_root / np.linalg.norm(post_root, axis=1, keepdims=True)
    root_error = np.linalg.norm(gt_post_root - post_root, axis=1)
    label = np.einsum("ni,ni->n", gt_post_root - post_root, post_ray)

    by_set: dict[str, dict[str, np.ndarray]] = {}
    for set_name, joint_ids in CORE_JOINT_SETS.items():
        raw = np.full(len(records), np.nan, dtype=np.float64)
        median_gap = np.full(len(records), np.inf, dtype=np.float64)
        max_gap = np.full(len(records), np.inf, dtype=np.float64)
        median_sine = np.zeros(len(records), dtype=np.float64)
        min_sine = np.zeros(len(records), dtype=np.float64)
        mad = np.full(len(records), np.inf, dtype=np.float64)
        valid_count = np.zeros(len(records), dtype=np.int64)

        for sample_index, sample in enumerate(frames_by_sample):
            camera_a = np.asarray(sample[1].gt_c2w, dtype=np.float64)
            camera_b = np.asarray(sample[2].gt_c2w, dtype=np.float64)
            origin_a, origin_b = camera_a[:3, 3], camera_b[:3, 3]
            rotation_a, rotation_b = camera_a[:3, :3], camera_b[:3, :3]
            candidates, gaps, sines = [], [], []
            for joint_id in joint_ids:
                local_a = pred[sample_index, 1, joint_id]
                local_b = pred[sample_index, 2, joint_id]
                if not (np.isfinite(local_a).all() and np.isfinite(local_b).all()):
                    continue
                direction_a = rotation_a @ (local_a / max(np.linalg.norm(local_a), 1e-12))
                direction_b = rotation_b @ (local_b / max(np.linalg.norm(local_b), 1e-12))
                midpoint, depth_a, depth_b, gap, sine = closest_rays(
                    origin_a, direction_a, origin_b, direction_b
                )
                if depth_a <= 0.0 or depth_b <= 0.0 or sine <= 1e-5:
                    continue
                relative_b_world = rotation_b @ (local_b - post_root[sample_index])
                candidate_root_world = midpoint - relative_b_world
                candidate_root_local = rotation_b.T @ (candidate_root_world - origin_b)
                delta = float(np.dot(candidate_root_local - post_root[sample_index], post_ray[sample_index]))
                if not np.isfinite(delta):
                    continue
                candidates.append(delta)
                gaps.append(gap)
                sines.append(sine)
            if not candidates:
                continue
            candidates_array = np.asarray(candidates, dtype=np.float64)
            gaps_array = np.asarray(gaps, dtype=np.float64)
            sines_array = np.asarray(sines, dtype=np.float64)
            center = float(np.median(candidates_array))
            raw[sample_index] = center
            median_gap[sample_index] = float(np.median(gaps_array))
            max_gap[sample_index] = float(np.max(gaps_array))
            median_sine[sample_index] = float(np.median(sines_array))
            min_sine[sample_index] = float(np.min(sines_array))
            mad[sample_index] = float(np.median(np.abs(candidates_array - center)))
            valid_count[sample_index] = len(candidates)
        by_set[set_name] = {
            "raw": raw,
            "median_gap": median_gap,
            "max_gap": max_gap,
            "median_sine": median_sine,
            "min_sine": min_sine,
            "mad": mad,
            "valid_count": valid_count,
        }
    return {
        "name": name,
        "records": records,
        "labels": label,
        "pred_roots": post_root,
        "gt_roots": gt_post_root,
        "rays": post_ray,
        "root_errors": root_error,
        "by_set": by_set,
    }


def metrics(payload: dict[str, Any], action: np.ndarray) -> dict[str, Any]:
    class MetricPayload:
        pass
    proxy = MetricPayload()
    proxy.pred_roots = payload["pred_roots"]
    proxy.gt_roots = payload["gt_roots"]
    proxy.rays = payload["rays"]
    proxy.root_errors = payload["root_errors"]
    proxy.labels = payload["labels"]
    return corrected_root_metrics(proxy, action)


def action_for(evidence: dict[str, np.ndarray], policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = evidence["raw"]
    accepted = (
        np.isfinite(raw)
        & (evidence["valid_count"] >= int(policy["min_valid"]))
        & (evidence["median_gap"] <= float(policy["max_median_gap_m"]))
        & (evidence["mad"] <= float(policy["max_mad_m"]))
        & (evidence["median_sine"] >= float(policy["min_median_sine"]))
        & (np.abs(raw) >= float(policy["min_abs_raw_m"]))
    )
    action = np.where(accepted, np.clip(raw, -float(policy["cap_m"]), float(policy["cap_m"])), 0.0)
    return action, accepted


def enumerate_policies(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for joint_set, evidence in payload["by_set"].items():
        joint_count = len(CORE_JOINT_SETS[joint_set])
        valid_options = sorted(set((1, min(3, joint_count), max(1, (joint_count + 1) // 2))))
        for min_valid in valid_options:
            for max_gap in (0.025, 0.05, 0.10, 0.20, 0.40):
                for max_mad in (0.025, 0.05, 0.10, 0.20, 0.40):
                    for min_sine in (0.025, 0.05, 0.10, 0.20):
                        for min_abs in (0.0, 0.05, 0.10, 0.20):
                            for cap in (0.10, 0.20, 0.30, 0.50, 1.0, 2.0):
                                policy = {
                                    "joint_set": joint_set,
                                    "min_valid": min_valid,
                                    "max_median_gap_m": max_gap,
                                    "max_mad_m": max_mad,
                                    "min_median_sine": min_sine,
                                    "min_abs_raw_m": min_abs,
                                    "cap_m": cap,
                                }
                                action, accepted = action_for(evidence, policy)
                                row_metrics = metrics(payload, action)
                                rows.append(
                                    {
                                        **policy,
                                        "coverage": float(np.mean(accepted)),
                                        "metrics": row_metrics,
                                    }
                                )
    return rows


def per_source(payload: dict[str, Any], action: np.ndarray) -> dict[str, Any]:
    sources = np.asarray([record["source"] for record in payload["records"]])
    result = {}
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        subset = {
            key: value[indices]
            for key, value in payload.items()
            if key in ("labels", "pred_roots", "gt_roots", "rays", "root_errors")
        }
        result[source] = metrics(subset, action[indices])
    return result


def run_dev(args: argparse.Namespace) -> None:
    payload = build_probe_payload(
        "offset0", args.train_archive, args.data_root, torch.device(args.device), args.smpl_chunk
    )
    rows = enumerate_policies(payload)
    safe = [
        row for row in rows
        if row["coverage"] >= 0.20 and row["metrics"]["harm_over_5cm_rate"] <= 0.10
    ]
    pool = safe if safe else [row for row in rows if row["coverage"] >= 0.20]
    selected = min(pool, key=lambda row: row["metrics"]["root_error_m"]["mean"])
    evidence = payload["by_set"][selected["joint_set"]]
    action, accepted = action_for(evidence, selected)
    frozen = {
        "protocol": {
            "selection_split": "offset0 only",
            "confirm_split": "offset50 must not influence this file",
            "camera": "exact frozen camera in this controlled feasibility probe",
            "inputs": "predicted SMPL-X joint rays from frames boundary-1 and boundary",
            "gt_candidate_or_gate_use": "none",
            "person_action": "post-cut rigid translation along its current pelvis ray only",
        },
        "joint_names": [JOINT_NAMES[index] for index in CORE_JOINT_SETS[selected["joint_set"]]],
        "policy": {key: value for key, value in selected.items() if key not in ("metrics",)},
        "dev_metrics": selected["metrics"],
        "dev_noop": metrics(payload, np.zeros(len(payload["labels"]))),
        "dev_per_source": per_source(payload, action),
        "dev_accepted": int(accepted.sum()),
        "decision_gate": {
            "root_gain_min": 0.08,
            "harm_over_5cm_max": 0.10,
            "coverage_min": 0.20,
        },
    }
    frozen["dev_pass"] = bool(
        selected["metrics"]["relative_gain"] >= 0.08
        and selected["metrics"]["harm_over_5cm_rate"] <= 0.10
        and selected["coverage"] >= 0.20
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dev_all_policies.json").write_text(
        json.dumps(json_ready(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    policy_path = args.output_dir / "FROZEN_TRIANGULATION_POLICY_BEFORE_CONFIRM.json"
    policy_path.write_text(json.dumps(json_ready(frozen), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "dev_evidence.npz",
        label=payload["labels"], raw=evidence["raw"], action=action, accepted=accepted,
        median_gap=evidence["median_gap"], median_sine=evidence["median_sine"], mad=evidence["mad"],
    )
    print(json.dumps(json_ready(frozen), indent=2, sort_keys=True))


def run_confirm(args: argparse.Namespace) -> None:
    policy_path = args.policy or (args.output_dir / "FROZEN_TRIANGULATION_POLICY_BEFORE_CONFIRM.json")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Frozen policy required before confirm: {policy_path}")
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    if not frozen.get("dev_pass", False):
        raise RuntimeError("Development gate failed; confirm split must remain unopened")
    policy = frozen["policy"]
    payload = build_probe_payload(
        "offset50", args.heldout_archive, args.data_root, torch.device(args.device), args.smpl_chunk
    )
    evidence = payload["by_set"][policy["joint_set"]]
    action, accepted = action_for(evidence, policy)
    result = {
        "frozen_policy": policy,
        "noop": metrics(payload, np.zeros(len(payload["labels"]))),
        "locked_policy": metrics(payload, action),
        "coverage": float(np.mean(accepted)),
        "accepted": int(accepted.sum()),
        "per_source": per_source(payload, action),
    }
    locked = result["locked_policy"]
    result["confirm_pass"] = bool(
        locked["relative_gain"] >= 0.08
        and locked["harm_over_5cm_rate"] <= 0.10
        and result["coverage"] >= 0.20
        and all(row["mean_delta_m"] <= 0.0 for row in result["per_source"].values())
    )
    result["decision"] = (
        "PASS controlled exact-camera feasibility; next required test is the same frozen policy with B0 cameras."
        if result["confirm_pass"]
        else "NO-GO controlled triangulation; do not spend the frozen B0 evaluation split."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "confirm_results.json").write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "confirm_evidence.npz",
        label=payload["labels"], raw=evidence["raw"], action=action, accepted=accepted,
        median_gap=evidence["median_gap"], median_sine=evidence["median_sine"], mad=evidence["mad"],
    )
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if not str(output).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in the Movie3R workspace under /data")
    if args.phase == "dev":
        run_dev(args)
    else:
        run_confirm(args)


if __name__ == "__main__":
    main()
