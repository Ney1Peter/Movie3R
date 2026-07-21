#!/usr/bin/env python3
"""Search fixed physical fallback rules beyond the selected V24 rotation bridge."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_V24 = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v25_rotation_fallback_audit"
)
SELECTED = "safe_tiered_extension_vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_v15(root: Path) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if len(output) != 180:
        raise RuntimeError(f"Expected 180 V15 cases, got {len(output)}")
    return output


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


def metrics(errors: list[float], baseline: list[float], active: list[bool]) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    selected = np.asarray(active, dtype=bool)
    return {
        "active_count": int(selected.sum()),
        "mean_rotation_deg": float(values.mean()),
        "median_rotation_deg": float(np.median(values)),
        "p90_rotation_deg": float(np.quantile(values, 0.90)),
        "p95_rotation_deg": float(np.quantile(values, 0.95)),
        "catastrophic_rate": float(np.mean(values > 45.0)),
        "rescued_catastrophic_count": int(np.sum((base > 45.0) & (values <= 45.0))),
        "introduced_catastrophic_count": int(np.sum((base <= 45.0) & (values > 45.0))),
        "harmful_over_5deg_count": int(np.sum(values > base + 5.0)),
        "good_case_harmful_count": int(np.sum((base < 10.0) & (values > base + 5.0))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v15 = load_v15(args.v15_dir)
    v24 = {
        row["case_name"]: row
        for row in json.loads(args.v24_report.read_text(encoding="utf-8"))["cases"]
    }

    cases = []
    baseline_errors = []
    for name, wide in v15.items():
        current = v24[name]
        gt = np.asarray(wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        fixed = np.asarray(wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        selected = np.asarray(current["variants"][SELECTED]["transform"], dtype=np.float32)[
            :3, :3
        ]
        diagnostics = current["diagnostics"]
        accepted = bool(
            diagnostics["trigger_safe_large_residual"]
            or diagnostics["trigger_safe_consensus"]
            or diagnostics["trigger_safe_low_texture_conflict"]
        )
        windows = {}
        for window_name, window in wide["windows"].items():
            metric = window["candidates"]["metric_full_full"]
            windows[window_name] = {
                "rotation": np.asarray(
                    window["candidates"]["coarse"]["transform"], dtype=np.float32
                )[:3, :3],
                "spread_deg": float(window["rotation_consensus"]["spread_deg"]),
                "metric_rotation": np.asarray(metric["transform"], dtype=np.float32)[:3, :3],
                "fit_residual_median_m": float(metric["fit_residual_median_m"]),
                "epipolar_median_px": float(metric["epipolar_median_px"]),
                "robust_inlier_ratio": float(metric["robust_inlier_ratio"]),
                "correspondence_count": int(metric["correspondence_count"]),
            }
        baseline_error = angle_deg(selected, gt)
        baseline_errors.append(baseline_error)
        cases.append(
            {
                "case_name": name,
                "source": current["source"],
                "gt": gt,
                "fixed": fixed,
                "selected": selected,
                "baseline_error": baseline_error,
                "texture": float(wide["texture_score"]),
                "torso_residual_deg": float(diagnostics["torso_residual_deg"]),
                "v24_accepted": accepted,
                "windows": windows,
            }
        )

    candidates = []
    for window_name in (
        "full_rgb_3p3",
        "background_only_1p1",
        "background_only_3p3",
    ):
        for spread_bound in (5.0, 10.0, 15.0, 20.0):
            for cap_deg in (15.0, 25.0, 35.0, 45.0, 60.0):
                errors = []
                active = []
                active_names = []
                for case in cases:
                    target = case["windows"][window_name]["rotation"]
                    target_residual = angle_deg(target, case["fixed"])
                    trigger = bool(
                        not case["v24_accepted"]
                        and case["torso_residual_deg"] >= 30.0
                        and case["windows"]["full_rgb_1p1"]["spread_deg"] > 15.0
                        and case["windows"][window_name]["spread_deg"] <= spread_bound
                        and target_residual <= 100.0
                        and target_residual >= case["torso_residual_deg"] + 5.0
                    )
                    rotation = (
                        capped_rotation(case["selected"], target, cap_deg)
                        if trigger
                        else case["selected"]
                    )
                    errors.append(angle_deg(rotation, case["gt"]))
                    active.append(trigger)
                    if trigger:
                        active_names.append(case["case_name"])
                result = metrics(errors, baseline_errors, active)
                candidates.append(
                    {
                        "family": "rejected_large_torso_alternate_window",
                        "window": window_name,
                        "spread_bound_deg": spread_bound,
                        "cap_deg": cap_deg,
                        "active_cases": active_names,
                        **result,
                    }
                )

    for texture_bound in (0.015, 0.020, 0.025, 0.030, 0.050):
        for spread_bound in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            for minimum_residual in (20.0, 30.0, 40.0, 50.0):
                for cap_deg in (15.0, 25.0, 35.0, 45.0, 60.0):
                    errors = []
                    active = []
                    active_names = []
                    for case in cases:
                        target = case["windows"]["full_rgb_1p1"]["rotation"]
                        target_residual = angle_deg(target, case["fixed"])
                        trigger = bool(
                            not case["v24_accepted"]
                            and case["torso_residual_deg"] < 10.0
                            and case["texture"] < texture_bound
                            and case["windows"]["full_rgb_1p1"]["spread_deg"] <= spread_bound
                            and minimum_residual <= target_residual <= 100.0
                        )
                        rotation = (
                            capped_rotation(case["selected"], target, cap_deg)
                            if trigger
                            else case["selected"]
                        )
                        errors.append(angle_deg(rotation, case["gt"]))
                        active.append(trigger)
                        if trigger:
                            active_names.append(case["case_name"])
                    result = metrics(errors, baseline_errors, active)
                    candidates.append(
                        {
                            "family": "low_torso_high_consensus_low_texture",
                            "texture_bound": texture_bound,
                            "spread_bound_deg": spread_bound,
                            "minimum_vggt_residual_deg": minimum_residual,
                            "cap_deg": cap_deg,
                            "active_cases": active_names,
                            **result,
                        }
                    )

    for spread_bound in (1.0, 2.0, 3.0, 5.0):
        for agreement_bound in (5.0, 10.0, 15.0, 20.0):
            for fit_bound in (0.40, 0.60, 0.80, 1.00):
                for cap_deg in (45.0, 60.0):
                    errors = []
                    active = []
                    active_names = []
                    for case in cases:
                        window = case["windows"]["full_rgb_1p1"]
                        target = window["rotation"]
                        target_residual = angle_deg(target, case["fixed"])
                        rotation_agreement = angle_deg(
                            target, window["metric_rotation"]
                        )
                        trigger = bool(
                            not case["v24_accepted"]
                            and case["torso_residual_deg"] < 10.0
                            and window["spread_deg"] <= spread_bound
                            and 30.0 <= target_residual <= 100.0
                            and rotation_agreement <= agreement_bound
                            and window["fit_residual_median_m"] <= fit_bound
                            and window["robust_inlier_ratio"] >= 0.50
                            and window["correspondence_count"] >= 100
                        )
                        rotation = (
                            capped_rotation(case["selected"], target, cap_deg)
                            if trigger
                            else case["selected"]
                        )
                        errors.append(angle_deg(rotation, case["gt"]))
                        active.append(trigger)
                        if trigger:
                            active_names.append(case["case_name"])
                    result = metrics(errors, baseline_errors, active)
                    candidates.append(
                        {
                            "family": "low_torso_camera_correspondence_consensus",
                            "spread_bound_deg": spread_bound,
                            "rotation_agreement_bound_deg": agreement_bound,
                            "fit_residual_median_bound_m": fit_bound,
                            "cap_deg": cap_deg,
                            "active_cases": active_names,
                            **result,
                        }
                    )

    candidates.sort(
        key=lambda row: (
            row["catastrophic_rate"],
            row["introduced_catastrophic_count"],
            row["good_case_harmful_count"],
            row["harmful_over_5deg_count"],
            row["mean_rotation_deg"],
        )
    )
    baseline = metrics(baseline_errors, baseline_errors, [False] * len(cases))
    safe = [
        row
        for row in candidates
        if row["active_count"] > 0
        and row["introduced_catastrophic_count"] == 0
        and row["good_case_harmful_count"] == 0
    ]
    report = {
        "experiment": "V25 fixed rotation-fallback rule audit",
        "scope": "GT is used only for evaluating fixed observable rules",
        "baseline_v24": baseline,
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe),
        "top_safe_candidates": safe[:100],
        "top_all_candidates": candidates[:100],
    }
    output = args.output_dir / "v25_rotation_fallback_rule_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"baseline": baseline, "top_safe": safe[:20]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
