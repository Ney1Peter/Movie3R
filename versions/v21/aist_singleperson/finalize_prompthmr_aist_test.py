#!/usr/bin/env python3
"""Seal disjoint PromptHMR AIST-CS150 test shards into one 100-case ledger.

The sharding rule is fixed in a hash-locked JSON file before the shards run.
This finalizer only reads those declared line intervals; it never chooses
between duplicate artifacts, filters low-scoring examples, or re-runs a
method.  It is intentionally strict: a missing case or invalid cache makes
the ledger fail instead of yielding a silently smaller denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import atomic_json, canonical_json_digest, sha256_file  # type: ignore


SCHEMA = "Bridge3R-AIST-PromptHMR-CS150-final-test-ledger-v1"
METHOD = "prompthmr_official"
FRAMES, CUT = 150, 74
METRICS = (
    "pa_mpjpe_body12_mm", "first_shot_anchor_mpjpe_body12_mm", "first_shot_anchor_root_error_mm",
    "first_shot_anchor_orientation_proxy_deg", "seam_root_excess_mm", "seam_orientation_excess_deg",
    "post_camera_relative_rotation_deg", "post_camera_relative_translation_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-lock", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty manifest: {path}")
    return values


def safe_name(case_id: str) -> str:
    return case_id.replace("/", "_")


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def metric_mean(row: dict[str, Any], metric: str) -> float:
    value = row.get("metrics", {}).get(metric, {}).get("mean")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric {metric} is absent/non-finite")
    return float(value)


def inspect_case(root: Path, case_id: str, runtime: dict[str, Any], evaluator: dict[str, Any], line: int) -> dict[str, Any]:
    name = safe_name(case_id)
    metrics_path = root / "metrics" / f"{name}.json"
    cache_path = root / "predictions" / f"{name}.npz"
    report_path = root / "predictions" / f"{name}.runtime.json"
    adapter_path = root / "predictions" / f"{name}.adapter.json"
    for path in (metrics_path, cache_path, report_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(f"declared shard case lacks required artifact: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    if metrics.get("case_id") != case_id or report.get("record") != runtime or report.get("methods") != [METHOD]:
        raise ValueError(f"case/protocol provenance mismatch at manifest line {line}")
    row = metrics.get("methods", {}).get(METHOD)
    if not isinstance(row, dict) or row.get("status") != "ok" or metrics.get("errors"):
        raise ValueError(f"external evaluator did not complete successfully at line {line}")
    if adapter.get("case_id") != case_id or adapter.get("runtime_gt_access") is not False:
        raise ValueError(f"adapter provenance or GT-access contract failed at line {line}")
    with np.load(cache_path, allow_pickle=False) as archive:
        prefix = METHOD + "__"
        required = ("cameras_c2w", "joints_world", "persistent_ids", "valid")
        missing = [prefix + key for key in required if prefix + key not in archive.files]
        if missing:
            raise KeyError(f"cache misses {missing} at line {line}")
        cameras = np.asarray(archive[prefix + "cameras_c2w"], dtype=np.float64)
        joints = np.asarray(archive[prefix + "joints_world"], dtype=np.float64)
        ids = np.asarray(archive[prefix + "persistent_ids"], dtype=np.int64)
        valid = np.asarray(archive[prefix + "valid"], dtype=bool)
    if cameras.shape != (FRAMES, 4, 4) or joints.ndim != 4 or joints.shape[0] != FRAMES or joints.shape[2:] != (24, 3) or ids.shape != valid.shape or ids.shape != joints.shape[:2]:
        raise ValueError(f"AIST cache shape drift at line {line}")
    track = row.get("track", {}); chosen = track.get("chosen_id")
    if not isinstance(chosen, int):
        raise ValueError(f"evaluator did not select a native persistent track at line {line}")
    selected = np.any(valid & (ids == chosen), axis=1)
    pre, post = int(selected[: CUT + 1].sum()), int(selected[CUT + 1 :].sum())
    if pre < 3 or post < 3 or not np.isfinite(joints[selected]).all():
        raise ValueError(f"cross-cut native track support/geometry failed at line {line}: pre={pre}, post={post}")
    if not np.isfinite(cameras).all():
        raise ValueError(f"PromptHMR camera cache is non-finite at line {line}")
    rotations = cameras[:, :3, :3]; identity = np.eye(3)
    ortho = float(np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - identity)))
    determinant = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if ortho >= 2e-3 or determinant >= 2e-3:
        raise ValueError(f"PromptHMR camera is not SE(3)-valid at line {line}")
    values = {metric: metric_mean(row, metric) for metric in METRICS}
    coverage = row.get("coverage", {})
    if float(coverage.get("valid_frame_coverage", -1.0)) != 1.0 or float(coverage.get("completion", -1.0)) != 1.0:
        raise ValueError(f"unexpected coverage/completion at line {line}")
    return {
        "line": line, "case_id": case_id, "source_root": str(root),
        "metric": str(metrics_path), "metric_sha256": sha256(metrics_path),
        "cache": str(cache_path), "cache_sha256": sha256(cache_path),
        "runtime_report": str(report_path), "runtime_report_sha256": sha256(report_path),
        "adapter": str(adapter_path), "adapter_sha256": sha256(adapter_path),
        "track": {"native_id": chosen, "pre_valid_frames": pre, "post_valid_frames": post},
        "camera": {"orthogonality_max": ortho, "determinant_error_max": determinant},
        "metrics": values,
    }


def build_tex(summary: dict[str, Any]) -> str:
    values = summary["metrics"]
    text = lambda name, digits=1: f"{values[name]['case_macro_mean']:.{digits}f}"
    return "\n".join([
        "% Auto-generated from the fixed 100-case AIST++ CS150 PromptHMR ledger; do not hand-edit.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & PA-MPJPE $\\downarrow$ & Anchor-MPJPE $\\downarrow$ & Seam-root $\\downarrow$ & Seam-orient. $\\downarrow$ & Rel. cam. rot. $\\downarrow$ & Coverage $\\uparrow$ \\\\",
        "\\midrule",
        f"PromptHMR (official, offline) & {text('pa_mpjpe_body12_mm')} & {text('first_shot_anchor_mpjpe_body12_mm')} & {text('seam_root_excess_mm')} & {text('seam_orientation_excess_deg')} & {text('post_camera_relative_rotation_deg')} & 100.0 \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"final test ledger refuses to overwrite {output}")
    runtime_path, evaluator_path, lock_path = args.runtime_manifest.resolve(), args.evaluator_manifest.resolve(), args.partition_lock.resolve()
    runtime, evaluator, lock = read_jsonl(runtime_path), read_jsonl(evaluator_path), json.loads(lock_path.read_text(encoding="utf-8"))
    if len(runtime) != len(evaluator) or [row.get("case_id") for row in runtime] != [row.get("case_id") for row in evaluator]:
        raise ValueError("runtime/evaluator CS150 manifests are not aligned")
    if lock.get("schema_version") != "Bridge3R-AIST-PromptHMR-test-partition-lock-v1" or lock.get("method") != METHOD:
        raise ValueError("unexpected partition lock")
    if lock.get("runtime_manifest_sha256") != sha256_file(runtime_path) or lock.get("evaluator_manifest_sha256") != sha256_file(evaluator_path):
        raise ValueError("partition lock does not match frozen manifests")
    intervals = lock.get("case_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("partition lock has no intervals")
    mapping: dict[int, Path] = {}
    for interval in intervals:
        start, stop = interval.get("inclusive_lines", [None, None])
        root = Path(str(interval.get("output_root", ""))).resolve()
        if not isinstance(start, int) or not isinstance(stop, int) or start < 1 or stop > len(runtime) or start > stop or not root.is_dir():
            raise ValueError(f"invalid declared test interval: {interval}")
        for line in range(start, stop + 1):
            if line in mapping:
                raise ValueError(f"overlapping partition at line {line}")
            mapping[line] = root
    if sorted(mapping) != list(range(1, len(runtime) + 1)):
        raise ValueError("declared intervals do not cover the full frozen test manifest exactly once")
    records = [inspect_case(mapping[line], str(runtime[line - 1]["case_id"]), runtime[line - 1], evaluator[line - 1], line) for line in range(1, len(runtime) + 1)]
    values = {metric: [record["metrics"][metric] for record in records] for metric in METRICS}
    summary = {
        "schema_version": SCHEMA, "method": METHOD, "formal_manifest_case_count": len(runtime), "reported_case_count": len(records),
        "partition_lock": str(lock_path), "partition_lock_sha256": sha256(lock_path),
        "runtime_manifest": str(runtime_path), "runtime_manifest_sha256": sha256_file(runtime_path),
        "evaluator_manifest": str(evaluator_path), "evaluator_manifest_sha256": sha256_file(evaluator_path),
        "records": records,
        "metrics": {metric: {"case_count": len(value), "case_macro_mean": mean(value), "case_macro_median": float(np.median(np.asarray(value, dtype=np.float64)))} for metric, value in values.items()},
        "coverage": {"case_macro_mean": 1.0, "completion_case_macro_mean": 1.0},
        "evaluation_contract": "Every declared test case retains one native track with >=3 valid frames before and after the hidden cut; each case contributes exactly one within-case metric mean; final scores are unweighted case macro means.",
        "selection_policy": lock["selection_rule"],
    }
    summary["content_sha256"] = canonical_json_digest(summary)
    output.mkdir(parents=True)
    atomic_json(output / "aggregate.json", summary)
    (output / "table.tex").write_text(build_tex(summary), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(records), "method": METHOD}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
