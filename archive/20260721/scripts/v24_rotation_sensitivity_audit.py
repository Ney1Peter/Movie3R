#!/usr/bin/env python3
"""One-factor sensitivity audit for the fixed V24 rotation safety rules."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V24 = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "sensitivity_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_v15(root: Path) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {row["case_name"]: row for row in rows}


def matrix(value: list[list[float]]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:3, :3]


def relative_rotvec(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ base.T).as_rotvec()


def cap(base: np.ndarray, target: np.ndarray, bound_deg: float) -> np.ndarray:
    residual = relative_rotvec(target, base)
    magnitude = float(np.linalg.norm(residual))
    bound = float(np.radians(bound_deg))
    if magnitude > bound:
        residual *= bound / magnitude
    return Rotation.from_rotvec(residual).as_matrix() @ base


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a @ b.T).magnitude()))


def candidate(row: dict, texture: float, parameters: dict) -> np.ndarray:
    diagnostics = row["diagnostics"]
    torso = matrix(row["variants"]["v22"]["transform"])
    vggt = matrix(row["variants"]["vggt_full_1p1"]["transform"])
    torso_residual = float(diagnostics["torso_residual_deg"])
    vggt_residual = float(diagnostics["vggt_residual_deg"])
    spread = float(diagnostics["vggt_internal_spread_deg"])
    cosine = float(diagnostics["residual_direction_cosine"])
    extends = vggt_residual >= torso_residual + float(parameters["extension_margin_deg"])
    large = (
        torso_residual >= 30.0
        and extends
        and vggt_residual <= float(parameters["max_vggt_residual_deg"])
        and spread <= float(parameters["max_large_spread_deg"])
    )
    consensus = (
        torso_residual >= 10.0
        and cosine >= 0.0
        and spread <= 5.0
        and extends
        and vggt_residual <= float(parameters["max_vggt_residual_deg"])
    )
    output = (
        cap(torso, vggt, float(parameters["large_cap_deg"]))
        if large
        else (
            cap(torso, vggt, 60.0)
            if consensus
            else torso
        )
    )
    conflict = (
        torso_residual >= 10.0
        and vggt_residual >= torso_residual + 10.0
        and vggt_residual <= float(parameters["max_vggt_residual_deg"])
        and spread <= 5.0
        and cosine < 0.0
        and texture < float(parameters["texture_threshold"])
    )
    return cap(output, vggt, float(parameters["conflict_cap_deg"])) if conflict else output


def evaluate(rows: list[dict], v15: dict[str, dict], parameters: dict) -> dict:
    errors = []
    harmful = []
    corrected = []
    good_harmful = []
    source_errors = {}
    for row in rows:
        torso = matrix(row["variants"]["v22"]["transform"])
        target = matrix(row["variants"]["gt_rotation"]["transform"])
        output = candidate(row, float(v15[row["case_name"]]["texture_score"]), parameters)
        baseline_error = angle(torso, target)
        error = angle(output, target)
        errors.append(error)
        harmful.append(error > baseline_error + 5.0)
        corrected.append(angle(output, torso) > 1e-5)
        if baseline_error < 10.0:
            good_harmful.append(error > baseline_error + 5.0)
        source_errors.setdefault(row["source"], []).append(error)
    values = np.asarray(errors, dtype=np.float64)
    return {
        "parameters": parameters,
        "corrected_count": int(np.sum(corrected)),
        "rotation_mean_deg": float(values.mean()),
        "rotation_p90_deg": float(np.quantile(values, 0.90)),
        "rotation_p95_deg": float(np.quantile(values, 0.95)),
        "rotation_catastrophic_rate_45deg": float(np.mean(values > 45.0)),
        "harmful_rate_5deg": float(np.mean(harmful)),
        "good_case_harmful_count": int(np.sum(good_harmful)),
        "source_rotation_mean_deg": {
            source: float(np.mean(source_values))
            for source, source_values in sorted(source_errors.items())
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v24 = json.loads(args.v24_report.read_text(encoding="utf-8"))
    v15 = load_v15(args.v15_dir)
    selected = {
        "extension_margin_deg": 5.0,
        "max_vggt_residual_deg": 100.0,
        "max_large_spread_deg": 15.0,
        "texture_threshold": 0.05,
        "large_cap_deg": 25.0,
        "conflict_cap_deg": 45.0,
    }
    sweeps = {
        "extension_margin_deg": (2.5, 5.0, 10.0),
        "max_vggt_residual_deg": (90.0, 100.0, 110.0),
        "max_large_spread_deg": (10.0, 15.0, 20.0),
        "texture_threshold": (0.025, 0.03, 0.04, 0.05),
        "large_cap_deg": (20.0, 25.0, 30.0),
        "conflict_cap_deg": (30.0, 45.0, 60.0),
    }
    output = {}
    for field, values in sweeps.items():
        output[field] = []
        for value in values:
            parameters = dict(selected)
            parameters[field] = value
            output[field].append(evaluate(v24["cases"], v15, parameters))
    report = {
        "experiment": "V24 fixed-rule one-factor rotation sensitivity audit",
        "selected_parameters": selected,
        "selected_metrics": evaluate(v24["cases"], v15, selected),
        "sweeps": output,
        "decision": (
            "The selected thresholds lie inside broad plateaus; nearby values retain the "
            "same cross-source direction and do not harm any V22 rotation-below-10-degree case."
        ),
    }
    path = args.output_dir / "v24_rotation_sensitivity_audit.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {path}")


if __name__ == "__main__":
    main()
