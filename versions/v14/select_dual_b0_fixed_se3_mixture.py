#!/usr/bin/env python3
"""Test one global, causal SE(3) mixture of the two existing B0 proposals.

This deliberately does *not* learn a selector, use per-example GT at runtime,
or change state ownership.  It probes the smallest remaining possibility in
the dual-proposal branch: a single alpha shared by every cut,

    B(alpha) = interpolate_SE3(B_cross96, B_old_adapted, alpha).

Alpha is selected only on the pair-disjoint development set.  A non-zero
candidate may be opened on confirmation only when it improves the aggregate
and tail without worsening the mean for any source.  Otherwise the correct
scientific outcome is a No-Go: reweighting existing implicit proposals does
not repair the camera tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV = REPO_ROOT / "output/v14_cut_first_cross_source/dual_b0_camera_features_dev96/report.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/dual_b0_fixed_se3_mixture"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
ALPHAS = tuple(np.linspace(0.0, 1.0, 9).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--dev-report", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--confirm-report", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Numerically stable rotation-vector logarithm, including near pi."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * .5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-5:
        # The diagonal formula gives a stable axis when sin(angle) vanishes.
        diagonal = np.maximum((np.diag(rotation) + 1.0) * .5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-8:
            for index in range(3):
                if index != largest:
                    axis[index] = (rotation[largest, index] + rotation[index, largest]) / (4.0 * axis[largest])
        norm = float(np.linalg.norm(axis))
        return angle * axis / max(norm, 1e-12)
    return angle / (2.0 * np.sin(angle)) * np.asarray(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]),
        dtype=np.float64,
    )


def so3_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64) + skew(vector)
    axis_hat = skew(vector / angle)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * (axis_hat @ axis_hat)


def interpolate_se3(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Rotation geodesic plus linear translation in the shared current gauge."""
    relative_rotation = left[:3, :3].T @ right[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = left[:3, :3] @ so3_exp(float(alpha) * so3_log(relative_rotation))
    result[:3, 3] = (1.0 - float(alpha)) * left[:3, 3] + float(alpha) * right[:3, 3]
    return result


def rotation_error_deg(camera: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(camera[:3, :3].T @ target[:3, :3]) - 1.0) * .5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def camera_metrics(camera: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation = float(np.linalg.norm(camera[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(camera, target)
    return {
        "translation_m": translation,
        "rotation_deg": rotation,
        "composite": translation + .02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def matrix(payload: Any, name: str) -> np.ndarray:
    value = np.asarray(payload, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return value


def evaluate(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        matrices = row.get("cameras_in_current_gauge")
        if matrices is None:
            raise ValueError("Report lacks cameras_in_current_gauge; rerun the dual-B0 cache with --overwrite")
        cross = matrix(matrices["cross96_b0"], "cross96_b0")
        old = matrix(matrices["old_b0_adapted"], "old_b0_adapted")
        target = matrix(matrices["target_evaluation_only"], "target_evaluation_only")
        evaluated.append({"source": row["source"], "metrics": camera_metrics(interpolate_se3(cross, old, alpha), target)})

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        composite = np.asarray([item["metrics"]["composite"] for item in items], dtype=np.float64)
        translation = np.asarray([item["metrics"]["translation_m"] for item in items], dtype=np.float64)
        rotation = np.asarray([item["metrics"]["rotation_deg"] for item in items], dtype=np.float64)
        catastrophic = np.asarray([item["metrics"]["catastrophic"] for item in items], dtype=bool)
        return {
            "count": int(len(items)),
            "translation_m": {"mean": float(translation.mean()), "p95": float(np.quantile(translation, .95))},
            "rotation_deg": {"mean": float(rotation.mean()), "p95": float(np.quantile(rotation, .95))},
            "composite": {"mean": float(composite.mean()), "p95": float(np.quantile(composite, .95))},
            "catastrophic_count": int(catastrophic.sum()),
        }

    return {
        "alpha": float(alpha), "overall": summary(evaluated),
        "by_source": {source: summary([item for item in evaluated if item["source"] == source]) for source in SOURCES},
    }


def qualifies(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    """Predeclared, conservative requirement before spending confirmation data."""
    overall, base = candidate["overall"], baseline["overall"]
    checks = {
        "nonzero_alpha": candidate["alpha"] > 0.0,
        "aggregate_gain_at_least_1pct": overall["composite"]["mean"] <= .99 * base["composite"]["mean"],
        "p95_noninferior": overall["composite"]["p95"] <= base["composite"]["p95"] + 1e-12,
        "catastrophic_reduction_at_least_10pct": overall["catastrophic_count"] <= .90 * base["catastrophic_count"],
        "every_source_mean_noninferior": all(
            candidate["by_source"][source]["composite"]["mean"] <= baseline["by_source"][source]["composite"]["mean"] + 1e-12
            for source in SOURCES
        ),
        "every_source_catastrophic_noninferior": all(
            candidate["by_source"][source]["catastrophic_count"] <= baseline["by_source"][source]["catastrophic_count"]
            for source in SOURCES
        ),
    }
    return all(checks.values()), checks


def rank(item: dict[str, Any]) -> tuple[float, float, float, float]:
    overall = item["result"]["overall"]
    return (overall["composite"]["mean"], overall["composite"]["p95"], overall["catastrophic_count"], item["alpha"])


def dev(args: argparse.Namespace) -> None:
    report = json.loads(args.dev_report.read_text(encoding="utf-8"))
    if report.get("failures"):
        raise RuntimeError("Development report has forward failures")
    baseline = evaluate(report["cases"], 0.0)
    candidates = []
    for alpha in ALPHAS:
        result = evaluate(report["cases"], alpha)
        qualified, checks = qualifies(result, baseline)
        candidates.append({"alpha": float(alpha), "result": result, "qualified": qualified, "checks": checks})
    candidates.sort(key=rank)
    qualified = [item for item in candidates if item["qualified"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "dual B0 fixed global SE(3) mixture, pair-disjoint development",
        "input": str(args.dev_report), "checkpoint": report["checkpoints"],
        "runtime_constraint": "one global alpha; cross96/old predictions only; no GT, selector, future frame, or state change at runtime",
        "interpolation": "R=R_cross exp(alpha log(R_cross^T R_old)); t=(1-alpha)t_cross+alpha t_old",
        "baseline_alpha_0": baseline, "candidates_ranked": candidates,
        "qualified_count": len(qualified),
    }
    (args.output_dir / "DEV_FIXED_MIXTURE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not qualified:
        print("NO_GO_DUAL_B0_FIXED_SE3_MIXTURE")
        return
    winner = min(qualified, key=rank)
    policy = {
        "freeze_id": "DUAL_B0_FIXED_SE3_MIXTURE_V1_20260803",
        "status": "frozen_after_pair_disjoint_dev_before_confirmation",
        "method": "fixed global SE(3) interpolation of cross96 and adapted old B0",
        "alpha": winner["alpha"], "development": winner,
        "fallback": "exact cross96 B0", "future_frames": 0, "state_change": "none",
        "confirmation_status": "not_run",
    }
    (args.output_dir / "FROZEN_POLICY_BEFORE_CONFIRM.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner}, indent=2))


def confirm(args: argparse.Namespace) -> None:
    if args.confirm_report is None or args.policy is None:
        raise ValueError("confirm requires --confirm-report and --policy")
    report = json.loads(args.confirm_report.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if report.get("failures"):
        raise RuntimeError("Confirmation report has forward failures")
    baseline, result = evaluate(report["cases"], 0.0), evaluate(report["cases"], float(policy["alpha"]))
    qualified, checks = qualifies(result, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CONFIRMATION.json").write_text(json.dumps({
        "experiment": "dual B0 fixed global SE(3) mixture confirmation", "input": str(args.confirm_report),
        "policy": str(args.policy), "baseline_alpha_0": baseline, "result": result,
        "qualified": qualified, "checks": checks,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"qualified": qualified, "checks": checks, "result": result}, indent=2))


def main() -> None:
    args = parse_args()
    dev(args) if args.phase == "dev" else confirm(args)


if __name__ == "__main__":
    main()
