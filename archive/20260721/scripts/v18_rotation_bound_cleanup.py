#!/usr/bin/env python3
"""V18 cleanup sweep for a single conservative V16 torso residual bound."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAM = REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v18_human_metric_translation" / "rotation_bound"
BOUNDS = (10.0, 15.0, 20.0, 25.0, 35.0, 45.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_cases(root: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v18_stream_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    if len(rows) != 180:
        raise RuntimeError(f"Expected 180 stream cases, got {len(rows)}")
    return rows


def rotation_error(estimated: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix((estimated @ target.T).astype(np.float64)).magnitude()))


def bounded_rotation(fixed: np.ndarray, torso: np.ndarray, maximum_deg: float) -> np.ndarray:
    delta = torso @ fixed.T
    vector = Rotation.from_matrix(delta.astype(np.float64)).as_rotvec()
    maximum = np.radians(maximum_deg)
    norm = float(np.linalg.norm(vector))
    if norm > maximum:
        vector *= maximum / norm
    return (Rotation.from_rotvec(vector).as_matrix() @ fixed).astype(np.float32)


def run_case(case: dict) -> dict:
    with np.load(case["cache_path"]) as cache:
        predicted_pose = cache["new_pose"].astype(np.float32)
        target_pose = cache["target_pose"].astype(np.float32)
        fixed = cache["fixed_transform"].astype(np.float32)
        torso = cache["torso_transform"].astype(np.float32)
    fixed_camera = fixed[:3, :3] @ predicted_pose[:3, :3]
    torso_camera = torso[:3, :3] @ predicted_pose[:3, :3]
    target_camera = target_pose[:3, :3]
    fixed_error = rotation_error(fixed_camera, target_camera)
    variants = {}
    for bound in BOUNDS:
        boundary_rotation = bounded_rotation(fixed[:3, :3], torso[:3, :3], bound)
        camera_rotation = boundary_rotation @ predicted_pose[:3, :3]
        variants[str(int(bound))] = {
            "rotation_error_deg": rotation_error(camera_rotation, target_camera),
            "applied_residual_deg": rotation_error(boundary_rotation, fixed[:3, :3]),
        }
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "fixed_rotation_error_deg": fixed_error,
        "raw_torso_rotation_error_deg": rotation_error(torso_camera, target_camera),
        "variants": variants,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate(cases: list[dict]) -> dict:
    output = {
        "fixed": {
            "rotation_deg": distribution([case["fixed_rotation_error_deg"] for case in cases]),
            "rotation_catastrophic_rate": float(np.mean([case["fixed_rotation_error_deg"] > 30.0 for case in cases])),
        }
    }
    for bound in BOUNDS:
        key = str(int(bound))
        errors = np.asarray([case["variants"][key]["rotation_error_deg"] for case in cases])
        fixed = np.asarray([case["fixed_rotation_error_deg"] for case in cases])
        output[key] = {
            "rotation_deg": distribution(errors.tolist()),
            "rotation_catastrophic_rate": float(np.mean(errors > 30.0)),
            "harmful_rate": float(np.mean(errors > fixed + 1.0)),
            "helpful_rate": float(np.mean(errors < fixed - 1.0)),
            "false_correction_rate_fixed_lt10": float(np.mean((fixed < 10.0) & (errors > fixed + 1.0))),
            "gain_fixed_gt30_deg": float(np.mean(fixed[fixed > 30.0] - errors[fixed > 30.0])) if np.any(fixed > 30.0) else None,
            "gain_fixed_gt60_deg": float(np.mean(fixed[fixed > 60.0] - errors[fixed > 60.0])) if np.any(fixed > 60.0) else None,
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V18 V16 Rotation-Bound Cleanup",
        "",
        "| Bound | Mean | P90 | P95 | R-cat | Helpful | Harmful | False on Fixed<10 | Gain >30 | Gain >60 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fixed = report["overall"]["fixed"]
    lines.append(
        f"| Fixed | {fixed['rotation_deg']['mean']:.2f} | {fixed['rotation_deg']['p90']:.2f} | "
        f"{fixed['rotation_deg']['p95']:.2f} | {100.0 * fixed['rotation_catastrophic_rate']:.1f}% | - | - | - | - | - |"
    )
    for bound in BOUNDS:
        row = report["overall"][str(int(bound))]
        lines.append(
            f"| {bound:.0f} | {row['rotation_deg']['mean']:.2f} | {row['rotation_deg']['p90']:.2f} | "
            f"{row['rotation_deg']['p95']:.2f} | {100.0 * row['rotation_catastrophic_rate']:.1f}% | "
            f"{100.0 * row['helpful_rate']:.1f}% | {100.0 * row['harmful_rate']:.1f}% | "
            f"{100.0 * row['false_correction_rate_fixed_lt10']:.1f}% | {row['gain_fixed_gt30_deg']:.2f} | "
            f"{row['gain_fixed_gt60_deg']:.2f} |"
        )
    if report["selection_satisfied"]:
        selection = f"Safety rule selected `{report['selected_bound_deg']:.0f} deg`."
    else:
        selection = (
            f"No bound satisfied the <=10% easy-sample false-correction rule; "
            f"`{report['selected_bound_deg']:.0f} deg` is the declared conservative fallback."
        )
    lines.extend(["", f"- {selection}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [run_case(case) for case in load_cases(args.stream_dir)]
    overall = aggregate(cases)
    by_source = {
        source: aggregate([case for case in cases if case["source"] == source])
        for source in sorted({case["source"] for case in cases})
    }
    # Predeclared safety rule: smallest bound retaining >=75% of the 45-degree large-error gain.
    gain45 = overall["45"]["gain_fixed_gt30_deg"]
    eligible = [
        bound
        for bound in BOUNDS
        if overall[str(int(bound))]["gain_fixed_gt30_deg"] >= 0.75 * gain45
        and overall[str(int(bound))]["false_correction_rate_fixed_lt10"] <= 0.10
    ]
    selected = min(eligible) if eligible else 20.0
    report = {
        "experiment": "V18 V16 Torso Residual Bound Cleanup",
        "case_count": len(cases),
        "protocol": {
            "bounds_deg": BOUNDS,
            "selection_rule": "smallest bound retaining >=75% of 45-degree gain on Fixed>30 and <=10% false correction on Fixed<10",
            "learned_selector_used": False,
        },
        "selected_bound_deg": selected,
        "selection_satisfied": bool(eligible),
        "overall": overall,
        "by_source": by_source,
        "cases": cases,
    }
    output = args.output_dir / "v18_rotation_bound_cleanup.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v18_rotation_bound_cleanup_summary.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
