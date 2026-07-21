#!/usr/bin/env python3
"""Validate emulated V24 and fixed V25 1+1 rotation rules on arbitrary caches."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V15 = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v15"
DEFAULT_V16 = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v16"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "evaluation"
V24_SELECTED = "safe_tiered_extension_vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--reference_v24_report", type=Path)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_shards(root: Path, pattern: str) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {row["case_name"]: row for row in rows}


def relative_rotvec(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix((target @ base.T).astype(np.float64)).as_rotvec()


def angle_deg(target: np.ndarray, base: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(relative_rotvec(target, base))))


def capped_rotation(base: np.ndarray, target: np.ndarray, cap_deg: float) -> np.ndarray:
    residual = relative_rotvec(target, base)
    magnitude = float(np.linalg.norm(residual))
    cap = float(np.radians(cap_deg))
    if magnitude > cap > 0.0:
        residual *= cap / magnitude
    return (Rotation.from_rotvec(residual).as_matrix() @ base).astype(np.float32)


def safe_gravity(v16: dict) -> tuple[np.ndarray, dict]:
    torso = np.asarray(
        v16["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"],
        dtype=np.float32,
    )[:3, :3]
    gravity_row = v16["fixed_candidates"]["fixed_torso_motion_gravity_1f_resolve_t"]
    gravity = np.asarray(gravity_row["transform"], dtype=np.float32)[:3, :3]
    old_frames = v16["ground_diagnostics"]["old"]["frames"]
    new_frame = v16["ground_diagnostics"]["new_1f"]["frames"][0]
    inlier = min(
        float(np.mean([frame["inlier_ratio"] for frame in old_frames])),
        float(new_frame["inlier_ratio"]),
    )
    alignment = min(
        float(np.mean([frame["reference_alignment"] for frame in old_frames])),
        float(new_frame["reference_alignment"]),
    )
    angle = float(gravity_row["gravity"]["bounded_residual_deg"])
    accepted = bool(angle >= 7.5 and inlier >= 0.5 and alignment >= 0.8)
    return (gravity if accepted else torso), {
        "accepted": accepted,
        "angle_deg": angle,
        "inlier_ratio": inlier,
        "reference_alignment": alignment,
    }


def v24_rotation(fixed: np.ndarray, torso: np.ndarray, v15: dict) -> tuple[np.ndarray, dict]:
    vggt = np.asarray(
        v15["windows"]["full_rgb_1p1"]["candidates"]["coarse"]["transform"],
        dtype=np.float32,
    )[:3, :3]
    torso_vector = relative_rotvec(torso, fixed)
    vggt_vector = relative_rotvec(vggt, fixed)
    torso_residual = float(np.degrees(np.linalg.norm(torso_vector)))
    vggt_residual = float(np.degrees(np.linalg.norm(vggt_vector)))
    direction_cosine = float(
        np.dot(torso_vector, vggt_vector)
        / max(np.linalg.norm(torso_vector) * np.linalg.norm(vggt_vector), 1e-9)
    )
    spread = float(
        v15["windows"]["full_rgb_1p1"]["rotation_consensus"]["spread_deg"]
    )
    extends = vggt_residual >= torso_residual + 5.0
    large = bool(
        torso_residual >= 30.0
        and extends
        and vggt_residual <= 100.0
        and spread <= 15.0
    )
    consensus = bool(
        torso_residual >= 10.0
        and direction_cosine >= 0.0
        and extends
        and spread <= 5.0
        and vggt_residual <= 100.0
    )
    base = (
        capped_rotation(torso, vggt, 25.0)
        if large
        else (capped_rotation(torso, vggt, 60.0) if consensus else torso)
    )
    low_texture_conflict = bool(
        torso_residual >= 10.0
        and vggt_residual >= torso_residual + 10.0
        and spread <= 5.0
        and direction_cosine < 0.0
        and float(v15["texture_score"]) < 0.05
        and vggt_residual <= 100.0
    )
    selected = capped_rotation(base, vggt, 45.0) if low_texture_conflict else base
    return selected, {
        "torso_residual_deg": torso_residual,
        "vggt_residual_deg": vggt_residual,
        "residual_direction_cosine": direction_cosine,
        "vggt_internal_spread_deg": spread,
        "vggt_extends_torso_by_5deg": extends,
        "trigger_safe_large_residual": large,
        "trigger_safe_consensus": consensus,
        "trigger_safe_low_texture_conflict": low_texture_conflict,
        "v24_accepted": bool(large or consensus or low_texture_conflict),
    }


def v25_rotation(
    fixed: np.ndarray,
    base: np.ndarray,
    diagnostics: dict,
    v15: dict,
) -> tuple[np.ndarray, dict]:
    full = v15["windows"]["full_rgb_1p1"]
    background = v15["windows"]["background_only_1p1"]
    background_rotation = np.asarray(
        background["candidates"]["coarse"]["transform"], dtype=np.float32
    )[:3, :3]
    background_residual = angle_deg(background_rotation, fixed)
    background_trigger = bool(
        not diagnostics["v24_accepted"]
        and diagnostics["torso_residual_deg"] >= 30.0
        and float(full["rotation_consensus"]["spread_deg"]) > 15.0
        and float(background["rotation_consensus"]["spread_deg"]) <= 15.0
        and background_residual <= 100.0
        and background_residual >= diagnostics["torso_residual_deg"] + 5.0
    )

    coarse = np.asarray(full["candidates"]["coarse"]["transform"], dtype=np.float32)[
        :3, :3
    ]
    metric_row = full["candidates"]["metric_full_full"]
    metric_rotation = np.asarray(metric_row["transform"], dtype=np.float32)[:3, :3]
    coarse_residual = angle_deg(coarse, fixed)
    camera_metric_agreement = angle_deg(coarse, metric_rotation)
    explicit_trigger = bool(
        not diagnostics["v24_accepted"]
        and diagnostics["torso_residual_deg"] < 10.0
        and float(full["rotation_consensus"]["spread_deg"]) <= 2.0
        and 30.0 <= coarse_residual <= 100.0
        and camera_metric_agreement <= 10.0
        and float(metric_row["fit_residual_median_m"]) <= 0.60
        and float(metric_row["robust_inlier_ratio"]) >= 0.50
        and int(metric_row["correspondence_count"]) >= 100
    )
    if explicit_trigger:
        selected = capped_rotation(base, coarse, 60.0)
    elif background_trigger:
        selected = capped_rotation(base, background_rotation, 60.0)
    else:
        selected = base
    return selected, {
        "trigger_background_1p1_fallback": background_trigger,
        "trigger_low_torso_explicit_consensus": explicit_trigger,
        "background_1p1_residual_deg": background_residual,
        "background_1p1_spread_deg": float(
            background["rotation_consensus"]["spread_deg"]
        ),
        "camera_metric_rotation_agreement_deg": camera_metric_agreement,
        "metric_fit_residual_median_m": float(metric_row["fit_residual_median_m"]),
        "metric_epipolar_median_px": float(metric_row["epipolar_median_px"]),
        "metric_robust_inlier_ratio": float(metric_row["robust_inlier_ratio"]),
        "metric_correspondence_count": int(metric_row["correspondence_count"]),
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def aggregate(rows: list[dict], method: str, baseline: str = "v24") -> dict:
    errors = np.asarray([row[method]["rotation_error_deg"] for row in rows])
    base = np.asarray([row[baseline]["rotation_error_deg"] for row in rows])
    return {
        "rotation_deg": distribution(errors.tolist()),
        "catastrophic_rate": float(np.mean(errors > 45.0)),
        "rescued_catastrophic_count": int(np.sum((base > 45.0) & (errors <= 45.0))),
        "introduced_catastrophic_count": int(np.sum((base <= 45.0) & (errors > 45.0))),
        "harmful_over_5deg_rate": float(np.mean(errors > base + 5.0)),
        "good_case_harmful_rate": float(np.mean((base < 10.0) & (errors > base + 5.0))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v15 = load_shards(args.v15_dir, "v15_candidates_shard_*_of_*.json")
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    names = sorted(set(v15) & set(v16))
    if not names or len(names) != len(v15) or len(names) != len(v16):
        raise RuntimeError(f"V15/V16 case mismatch: {len(v15)}/{len(v16)}/{len(names)}")
    reference = None
    if args.reference_v24_report:
        reference = {
            row["case_name"]: row
            for row in json.loads(args.reference_v24_report.read_text(encoding="utf-8"))["cases"]
        }

    rows = []
    for name in names:
        wide = v15[name]
        human = v16[name]
        fixed = np.asarray(wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        gt = np.asarray(wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        torso, gravity_diagnostics = safe_gravity(human)
        v24_value, v24_diagnostics = v24_rotation(fixed, torso, wide)
        v25_value, v25_diagnostics = v25_rotation(
            fixed, v24_value, v24_diagnostics, wide
        )
        row = {
            "case_name": name,
            "source": human["record"]["source"],
            "gravity": gravity_diagnostics,
            "v24_diagnostics": v24_diagnostics,
            "v25_diagnostics": v25_diagnostics,
            "fixed": {"rotation_error_deg": angle_deg(fixed, gt)},
            "torso_gravity": {"rotation_error_deg": angle_deg(torso, gt)},
            "v24": {"rotation_error_deg": angle_deg(v24_value, gt)},
            "v25": {"rotation_error_deg": angle_deg(v25_value, gt)},
        }
        if reference is not None:
            reference_rotation = np.asarray(
                reference[name]["variants"][V24_SELECTED]["transform"], dtype=np.float32
            )[:3, :3]
            row["v24_reference_difference_deg"] = angle_deg(v24_value, reference_rotation)
        rows.append(row)

    methods = ("fixed", "torso_gravity", "v24", "v25")
    reference_differences = [
        row["v24_reference_difference_deg"]
        for row in rows
        if "v24_reference_difference_deg" in row
    ]
    report = {
        "experiment": "V25 disjoint holdout 1+1 rotation validation",
        "case_count": len(rows),
        "protocol": {
            "thresholds_frozen_from_original_180": True,
            "holdout_used_for_tuning": False,
            "post_cut_frames": 1,
            "gt_runtime_information": False,
        },
        "trigger_counts": {
            "v24": int(sum(row["v24_diagnostics"]["v24_accepted"] for row in rows)),
            "background_1p1": int(
                sum(
                    row["v25_diagnostics"]["trigger_background_1p1_fallback"]
                    for row in rows
                )
            ),
            "low_torso_explicit_consensus": int(
                sum(
                    row["v25_diagnostics"]["trigger_low_torso_explicit_consensus"]
                    for row in rows
                )
            ),
        },
        "overall": {method: aggregate(rows, method) for method in methods},
        "comparisons": {
            "torso_gravity_vs_fixed": aggregate(
                rows, "torso_gravity", baseline="fixed"
            ),
            "v24_vs_fixed": aggregate(rows, "v24", baseline="fixed"),
            "v24_vs_torso_gravity": aggregate(
                rows, "v24", baseline="torso_gravity"
            ),
            "v25_vs_v24": aggregate(rows, "v25", baseline="v24"),
        },
        "by_source": {
            source: {
                method: aggregate(
                    [row for row in rows if row["source"] == source], method
                )
                for method in methods
            }
            for source in sorted({row["source"] for row in rows})
        },
        "v24_reference_reproduction": (
            None
            if not reference_differences
            else {
                "mean_difference_deg": float(np.mean(reference_differences)),
                "max_difference_deg": float(np.max(reference_differences)),
                "within_0_001deg_rate": float(
                    np.mean(np.asarray(reference_differences) <= 0.001)
                ),
            }
        ),
        "cases": rows,
    }
    output = args.output_dir / "v25_holdout_rotation_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("case_count", "trigger_counts", "overall", "v24_reference_reproduction")}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
