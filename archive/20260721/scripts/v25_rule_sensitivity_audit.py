#!/usr/bin/env python3
"""One-at-a-time sensitivity audit for the two V25 1+1 rotation fallbacks."""

from __future__ import annotations

import argparse
import glob
import json
from copy import deepcopy
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
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_explicit_consensus_bridge" / "sensitivity_audit"
SELECTED = "safe_tiered_extension_vggt"


BACKGROUND_DEFAULTS = {
    "full_spread_min_deg": 15.0,
    "background_spread_max_deg": 15.0,
    "residual_max_deg": 100.0,
    "extension_min_deg": 5.0,
    "cap_deg": 60.0,
}
EXPLICIT_DEFAULTS = {
    "torso_residual_max_deg": 10.0,
    "coarse_spread_max_deg": 2.0,
    "coarse_residual_min_deg": 30.0,
    "coarse_residual_max_deg": 100.0,
    "rotation_agreement_max_deg": 10.0,
    "fit_residual_median_max_m": 0.60,
    "robust_inlier_min": 0.50,
    "correspondence_min": 100,
    "cap_deg": 60.0,
}


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


def base_accepted(diagnostics: dict) -> bool:
    return bool(
        diagnostics["trigger_safe_large_residual"]
        or diagnostics["trigger_safe_consensus"]
        or diagnostics["trigger_safe_low_texture_conflict"]
    )


def prepare(v15: dict[str, dict], v24: dict[str, dict]) -> list[dict]:
    rows = []
    for name, wide in v15.items():
        current = v24[name]
        full = wide["windows"]["full_rgb_1p1"]
        background = wide["windows"]["background_only_1p1"]
        metric = full["candidates"]["metric_full_full"]
        fixed = np.asarray(wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        base = np.asarray(current["variants"][SELECTED]["transform"], dtype=np.float32)[
            :3, :3
        ]
        gt = np.asarray(wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        coarse = np.asarray(full["candidates"]["coarse"]["transform"], dtype=np.float32)[
            :3, :3
        ]
        background_rotation = np.asarray(
            background["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        metric_rotation = np.asarray(metric["transform"], dtype=np.float32)[:3, :3]
        rows.append(
            {
                "case_name": name,
                "source": current["source"],
                "fixed": fixed,
                "base": base,
                "gt": gt,
                "base_error": angle_deg(base, gt),
                "base_accepted": base_accepted(current["diagnostics"]),
                "torso_residual_deg": float(current["diagnostics"]["torso_residual_deg"]),
                "full_spread_deg": float(full["rotation_consensus"]["spread_deg"]),
                "background_spread_deg": float(
                    background["rotation_consensus"]["spread_deg"]
                ),
                "background_rotation": background_rotation,
                "background_residual_deg": angle_deg(background_rotation, fixed),
                "coarse": coarse,
                "coarse_residual_deg": angle_deg(coarse, fixed),
                "coarse_metric_agreement_deg": angle_deg(coarse, metric_rotation),
                "fit_residual_median_m": float(metric["fit_residual_median_m"]),
                "epipolar_median_px": float(metric["epipolar_median_px"]),
                "robust_inlier_ratio": float(metric["robust_inlier_ratio"]),
                "correspondence_count": int(metric["correspondence_count"]),
            }
        )
    return rows


def evaluate(rows: list[dict], background: dict, explicit: dict) -> dict:
    errors = []
    background_cases = []
    explicit_cases = []
    for row in rows:
        background_trigger = bool(
            not row["base_accepted"]
            and row["torso_residual_deg"] >= 30.0
            and row["full_spread_deg"] > background["full_spread_min_deg"]
            and row["background_spread_deg"] <= background["background_spread_max_deg"]
            and row["background_residual_deg"] <= background["residual_max_deg"]
            and row["background_residual_deg"]
            >= row["torso_residual_deg"] + background["extension_min_deg"]
        )
        explicit_trigger = bool(
            not row["base_accepted"]
            and row["torso_residual_deg"] < explicit["torso_residual_max_deg"]
            and row["full_spread_deg"] <= explicit["coarse_spread_max_deg"]
            and explicit["coarse_residual_min_deg"]
            <= row["coarse_residual_deg"]
            <= explicit["coarse_residual_max_deg"]
            and row["coarse_metric_agreement_deg"]
            <= explicit["rotation_agreement_max_deg"]
            and row["fit_residual_median_m"]
            <= explicit["fit_residual_median_max_m"]
            and row["robust_inlier_ratio"] >= explicit["robust_inlier_min"]
            and row["correspondence_count"] >= explicit["correspondence_min"]
        )
        if explicit_trigger:
            rotation = capped_rotation(row["base"], row["coarse"], explicit["cap_deg"])
            explicit_cases.append(row["case_name"])
        elif background_trigger:
            rotation = capped_rotation(
                row["base"], row["background_rotation"], background["cap_deg"]
            )
            background_cases.append(row["case_name"])
        else:
            rotation = row["base"]
        errors.append(angle_deg(rotation, row["gt"]))

    values = np.asarray(errors, dtype=np.float64)
    baseline = np.asarray([row["base_error"] for row in rows], dtype=np.float64)
    return {
        "mean_rotation_deg": float(values.mean()),
        "p90_rotation_deg": float(np.quantile(values, 0.90)),
        "p95_rotation_deg": float(np.quantile(values, 0.95)),
        "catastrophic_rate": float(np.mean(values > 45.0)),
        "rescued_catastrophic_count": int(np.sum((baseline > 45.0) & (values <= 45.0))),
        "introduced_catastrophic_count": int(np.sum((baseline <= 45.0) & (values > 45.0))),
        "harmful_over_5deg_count": int(np.sum(values > baseline + 5.0)),
        "good_case_harmful_count": int(np.sum((baseline < 10.0) & (values > baseline + 5.0))),
        "background_trigger_cases": background_cases,
        "explicit_trigger_cases": explicit_cases,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v15 = load_v15(args.v15_dir)
    v24 = {
        row["case_name"]: row
        for row in json.loads(args.v24_report.read_text(encoding="utf-8"))["cases"]
    }
    rows = prepare(v15, v24)
    baseline = evaluate(
        rows,
        {**BACKGROUND_DEFAULTS, "background_spread_max_deg": -1.0},
        {**EXPLICIT_DEFAULTS, "coarse_spread_max_deg": -1.0},
    )
    selected = evaluate(rows, BACKGROUND_DEFAULTS, EXPLICIT_DEFAULTS)

    sweeps = {}
    background_values = {
        "full_spread_min_deg": (10.0, 15.0, 20.0, 25.0),
        "background_spread_max_deg": (10.0, 15.0, 20.0, 25.0),
        "residual_max_deg": (90.0, 100.0, 110.0),
        "extension_min_deg": (0.0, 5.0, 10.0),
        "cap_deg": (45.0, 60.0),
    }
    for field, values in background_values.items():
        sweeps[f"background.{field}"] = []
        for value in values:
            current = deepcopy(BACKGROUND_DEFAULTS)
            current[field] = value
            sweeps[f"background.{field}"].append(
                {"value": value, **evaluate(rows, current, EXPLICIT_DEFAULTS)}
            )

    explicit_values = {
        "torso_residual_max_deg": (5.0, 10.0, 15.0),
        "coarse_spread_max_deg": (1.0, 2.0, 3.0, 5.0),
        "rotation_agreement_max_deg": (10.0, 15.0, 20.0),
        "fit_residual_median_max_m": (0.60, 0.80, 1.00),
        "robust_inlier_min": (0.40, 0.50, 0.60),
        "correspondence_min": (50, 100, 200),
        "cap_deg": (45.0, 60.0),
    }
    for field, values in explicit_values.items():
        sweeps[f"explicit.{field}"] = []
        for value in values:
            current = deepcopy(EXPLICIT_DEFAULTS)
            current[field] = value
            sweeps[f"explicit.{field}"].append(
                {"value": value, **evaluate(rows, BACKGROUND_DEFAULTS, current)}
            )

    all_rows = [item for values in sweeps.values() for item in values]
    stable = [
        row
        for row in all_rows
        if row["introduced_catastrophic_count"] == 0
        and row["harmful_over_5deg_count"] == 0
        and row["good_case_harmful_count"] == 0
    ]
    report = {
        "experiment": "V25 selected 1+1 rotation-rule sensitivity audit",
        "case_count": len(rows),
        "baseline_v24": baseline,
        "selected_v25": selected,
        "perturbation_count": len(all_rows),
        "stable_perturbation_count": len(stable),
        "stable_perturbation_rate": float(len(stable) / max(len(all_rows), 1)),
        "sweeps": sweeps,
    }
    output = args.output_dir / "v25_rule_sensitivity_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "selected": selected,
                "stable_perturbations": f"{len(stable)}/{len(all_rows)}",
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
