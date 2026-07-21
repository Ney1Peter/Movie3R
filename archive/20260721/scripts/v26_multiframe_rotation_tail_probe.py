#!/usr/bin/env python3
"""Probe a fixed 3+3 consensus fallback for the residual V25 rotation tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v25_holdout_rotation_validation import (
    aggregate,
    angle_deg,
    load_shards,
    safe_gravity,
    v24_rotation,
    v25_rotation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_V16 = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v26_multiframe_rotation_tail"
SCALES = (1.0, 1.25, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected_scale", type=float, default=1.25)
    parser.add_argument("--dataset_role", default="development")
    return parser.parse_args()


def relative_rotvec(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix((target @ base.T).astype(np.float64)).as_rotvec()


def extrapolated_rotation(
    fixed: np.ndarray,
    first: np.ndarray,
    third: np.ndarray,
    scale: float,
    cap_deg: float = 120.0,
) -> np.ndarray:
    residual = 0.5 * (
        relative_rotvec(first, fixed) + relative_rotvec(third, fixed)
    )
    residual *= float(scale)
    magnitude = float(np.linalg.norm(residual))
    cap = float(np.radians(cap_deg))
    if magnitude > cap:
        residual *= cap / magnitude
    return (Rotation.from_rotvec(residual).as_matrix() @ fixed).astype(np.float32)


def tail_candidate(
    fixed: np.ndarray,
    base: np.ndarray,
    v24_diagnostics: dict,
    v25_diagnostics: dict,
    wide: dict,
    scale: float,
) -> tuple[np.ndarray, dict]:
    first_window = wide["windows"]["full_rgb_1p1"]
    third_window = wide["windows"]["full_rgb_3p3"]
    first = np.asarray(
        first_window["candidates"]["coarse"]["transform"], dtype=np.float32
    )[:3, :3]
    third = np.asarray(
        third_window["candidates"]["coarse"]["transform"], dtype=np.float32
    )[:3, :3]
    first_vector = relative_rotvec(first, fixed)
    third_vector = relative_rotvec(third, fixed)
    first_residual = float(np.degrees(np.linalg.norm(first_vector)))
    third_residual = float(np.degrees(np.linalg.norm(third_vector)))
    vector_cosine = float(
        np.dot(first_vector, third_vector)
        / max(np.linalg.norm(first_vector) * np.linalg.norm(third_vector), 1e-9)
    )
    agreement = angle_deg(first, third)
    v25_accepted = bool(
        v25_diagnostics["trigger_background_1p1_fallback"]
        or v25_diagnostics["trigger_low_torso_explicit_consensus"]
    )
    trigger = bool(
        not v24_diagnostics["v24_accepted"]
        and not v25_accepted
        and v24_diagnostics["torso_residual_deg"] >= 30.0
        and float(first_window["rotation_consensus"]["spread_deg"]) > 15.0
        and float(third_window["rotation_consensus"]["spread_deg"]) <= 30.0
        and v24_diagnostics["residual_direction_cosine"] >= 0.80
        and v24_diagnostics["vggt_extends_torso_by_5deg"]
        and first_residual <= 100.0
        and third_residual >= v24_diagnostics["torso_residual_deg"] + 5.0
        and third_residual <= 100.0
        and agreement <= 15.0
        and vector_cosine >= 0.80
    )
    selected = (
        extrapolated_rotation(fixed, first, third, scale) if trigger else base
    )
    return selected, {
        "trigger": trigger,
        "scale": float(scale),
        "full_1p1_residual_deg": first_residual,
        "full_3p3_residual_deg": third_residual,
        "full_1p1_spread_deg": float(
            first_window["rotation_consensus"]["spread_deg"]
        ),
        "full_3p3_spread_deg": float(
            third_window["rotation_consensus"]["spread_deg"]
        ),
        "full_1p1_3p3_agreement_deg": agreement,
        "residual_direction_cosine": vector_cosine,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if float(args.selected_scale) not in SCALES:
        raise ValueError(f"selected_scale must be one of {SCALES}")
    v15 = load_shards(args.v15_dir, "v15_candidates_shard_*_of_*.json")
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    names = sorted(set(v15) & set(v16))
    if not names or len(names) != len(v15) or len(names) != len(v16):
        raise RuntimeError(f"V15/V16 case mismatch: {len(v15)}/{len(v16)}/{len(names)}")

    rows = []
    for name in names:
        wide, human = v15[name], v16[name]
        fixed = np.asarray(
            wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
        )[:3, :3]
        gt = np.asarray(
            wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32
        )[:3, :3]
        torso, _ = safe_gravity(human)
        v24_value, v24_diagnostics = v24_rotation(fixed, torso, wide)
        v25_value, v25_diagnostics = v25_rotation(
            fixed, v24_value, v24_diagnostics, wide
        )
        row = {
            "case_name": name,
            "source": human["record"]["source"],
            "v25": {"rotation_error_deg": angle_deg(v25_value, gt)},
            "variants": {},
        }
        for scale in SCALES:
            value, diagnostics = tail_candidate(
                fixed,
                v25_value,
                v24_diagnostics,
                v25_diagnostics,
                wide,
                scale,
            )
            row["variants"][str(scale)] = {
                "rotation_error_deg": angle_deg(value, gt),
                "diagnostics": diagnostics,
            }
        rows.append(row)

    report = {
        "experiment": "V26 fixed 3+3 multiframe rotation-tail probe",
        "case_count": len(rows),
        "dataset_role": args.dataset_role,
        "protocol": {
            "base": "V25 1+1 after GT-epipolar trigger removal",
            "post_cut_frames": 3,
            "fixed_latency_frames": 2,
            "gt_runtime_information": False,
            "selected_scale_frozen_before_holdout": float(args.selected_scale),
            "status": "exploratory; extrapolation requires independent validation",
        },
        "v25": aggregate(rows, "v25", baseline="v25"),
        "variants": {},
        "cases": rows,
    }
    for scale in SCALES:
        key = str(scale)
        flattened = [
            {
                "case_name": row["case_name"],
                "source": row["source"],
                "v25": row["v25"],
                key: row["variants"][key],
            }
            for row in rows
        ]
        report["variants"][key] = {
            "trigger_count": int(
                sum(row["variants"][key]["diagnostics"]["trigger"] for row in rows)
            ),
            "overall": aggregate(flattened, key, baseline="v25"),
            "by_source": {
                source: aggregate(
                    [row for row in flattened if row["source"] == source],
                    key,
                    baseline="v25",
                )
                for source in sorted({row["source"] for row in rows})
            },
            "active_cases": [
                row["case_name"]
                for row in rows
                if row["variants"][key]["diagnostics"]["trigger"]
            ],
        }
    report["selected"] = report["variants"][str(float(args.selected_scale))]
    output = args.output_dir / f"v26_multiframe_rotation_tail_{args.dataset_role}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "v25": report["v25"],
                "variants": report["variants"],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
