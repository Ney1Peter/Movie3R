#!/usr/bin/env python3
"""Check CUDA-FP32 against the audited CPU reference on a smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decisions(report: dict) -> dict:
    geometry = report["geometry"]
    return {
        "association_pairs": geometry["association"]["pairs"],
        "brtc_accept": [bool(row["accepted"]) for row in geometry["brtc"]["people"]],
        "brtc_pairs": [[row["pre_index"], row["post_index"]] for row in geometry["brtc"]["people"]],
        "brtc_lambda": geometry["brtc"]["selected_residual_lambda"],
        "c1_gates": geometry["c1"]["gates"],
        "adaptive": [[bool(row["accepted"]), row["reason"]] for row in geometry["adaptive"]],
        "detector_labels": report["runtime"]["causal_gru_detector"]["labels"],
    }


def main() -> None:
    a = args()
    rows = {}
    discrete_equal = True
    camera_max = vertex_max = joint_max = 0.0
    with np.load(a.cpu, allow_pickle=False) as cpu, np.load(a.cuda, allow_pickle=False) as cuda:
        if set(cpu.files) != set(cuda.files):
            raise ValueError("CPU/CUDA cache keys differ")
        for key in cpu.files:
            first, second = np.asarray(cpu[key]), np.asarray(cuda[key])
            if first.dtype.kind in "iub":
                equal = bool(np.array_equal(first, second))
                rows[key] = {"exact": equal}
                discrete_equal &= equal
                continue
            finite_same = bool(np.array_equal(np.isfinite(first), np.isfinite(second)))
            valid = np.isfinite(first) & np.isfinite(second)
            maximum = float(np.max(np.abs(first[valid] - second[valid]))) if valid.any() else 0.0
            rows[key] = {"finite_mask_exact": finite_same, "max_abs": maximum}
            discrete_equal &= finite_same
            if key.endswith("cameras_c2w"):
                camera_max = max(camera_max, maximum)
            elif key.endswith("vertices_world"):
                vertex_max = max(vertex_max, maximum)
            elif key.endswith("joints_world"):
                joint_max = max(joint_max, maximum)
    cpu_report = json.loads(a.cpu_report.read_text(encoding="utf-8"))
    cuda_report = json.loads(a.cuda_report.read_text(encoding="utf-8"))
    cpu_decisions, cuda_decisions = decisions(cpu_report), decisions(cuda_report)
    decision_equal = cpu_decisions == cuda_decisions
    thresholds = {"camera_max_abs": 1e-4, "vertex_max_abs_m": 1e-3, "joint_max_abs_m": 1e-3}
    passed = bool(
        discrete_equal and decision_equal
        and camera_max <= thresholds["camera_max_abs"]
        and vertex_max <= thresholds["vertex_max_abs_m"]
        and joint_max <= thresholds["joint_max_abs_m"]
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "policy": "CUDA FP32 is accepted only with exact discrete/gate decisions and bounded floating drift",
        "thresholds": thresholds,
        "summary": {
            "discrete_and_finite_masks_exact": discrete_equal,
            "decisions_exact": decision_equal,
            "camera_max_abs": camera_max,
            "vertex_max_abs_m": vertex_max,
            "joint_max_abs_m": joint_max,
        },
        "cpu_decisions": cpu_decisions,
        "cuda_decisions": cuda_decisions,
        "arrays": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), **report["summary"], "status": report["status"]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
