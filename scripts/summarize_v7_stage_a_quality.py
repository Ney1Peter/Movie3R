#!/usr/bin/env python3
"""Summarize and filter V7 Stage-A pseudo labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("output/v7_ms_aist_shot2_stage_a/stage_a_manifest.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min_raw_boundary_foot", type=float, default=0.15)
    parser.add_argument("--min_boundary_improve", type=float, default=0.05)
    parser.add_argument("--max_boundary_ratio", type=float, default=0.90)
    parser.add_argument("--max_settle_worse_abs", type=float, default=0.25)
    parser.add_argument("--max_post_pair_mean_worse_abs", type=float, default=0.05)
    parser.add_argument("--max_delta_t", type=float, default=1.25)
    parser.add_argument("--max_delta_r_deg", type=float, default=45.0)
    parser.add_argument("--no_require_tokens", action="store_true")
    return parser.parse_args()


def output_root_from_manifest(manifest_path: Path, manifest: dict) -> Path:
    root = Path(manifest.get("output_root", manifest_path.parent))
    if not root.is_absolute():
        root = manifest_path.parent / root
    return root.resolve()


def get_transition(summary: dict, key: str, metric: str) -> float:
    return float(summary[key][metric])


def evaluate_case(case_entry: dict, output_root: Path, args: argparse.Namespace) -> dict:
    name = case_entry["case"]["name"]
    reasons: list[str] = []
    metrics: dict[str, float | list[float] | bool] = {}

    if case_entry.get("status") != "ok":
        reasons.append(f"status_{case_entry.get('status', 'unknown')}")
    summary = case_entry.get("pseudo_summary")
    if summary is None:
        reasons.append("missing_pseudo_summary")
    else:
        raw_boundary = get_transition(summary, "raw_transition", "boundary_foot_jump")
        teacher_boundary = get_transition(summary, "corrected_transition", "boundary_foot_jump")
        raw_settle = get_transition(summary, "raw_transition", "settle_foot_jump")
        teacher_settle = get_transition(summary, "corrected_transition", "settle_foot_jump")
        raw_pair_mean = get_transition(summary, "raw_transition", "post_pair_foot_jump_mean")
        teacher_pair_mean = get_transition(summary, "corrected_transition", "post_pair_foot_jump_mean")
        delta_t = [float(x) for x in summary.get("delta_t_norm", [])]
        delta_r = [float(x) for x in summary.get("delta_rotvec_deg", [])]

        boundary_improve = raw_boundary - teacher_boundary
        boundary_ratio = teacher_boundary / max(raw_boundary, 1e-6)
        metrics.update(
            {
                "raw_boundary_foot": raw_boundary,
                "teacher_boundary_foot": teacher_boundary,
                "boundary_improve": boundary_improve,
                "boundary_ratio": boundary_ratio,
                "raw_settle_foot": raw_settle,
                "teacher_settle_foot": teacher_settle,
                "settle_worse_abs": teacher_settle - raw_settle,
                "raw_post_pair_mean": raw_pair_mean,
                "teacher_post_pair_mean": teacher_pair_mean,
                "post_pair_mean_worse_abs": teacher_pair_mean - raw_pair_mean,
                "delta_t_norm": delta_t,
                "delta_rotvec_deg": delta_r,
                "max_delta_t": max(delta_t) if delta_t else 0.0,
                "max_delta_r_deg": max(delta_r) if delta_r else 0.0,
            }
        )

        if raw_boundary < args.min_raw_boundary_foot:
            reasons.append("raw_boundary_below_min")
        if boundary_improve < args.min_boundary_improve and boundary_ratio > args.max_boundary_ratio:
            reasons.append("boundary_not_improved")
        if teacher_settle > raw_settle + args.max_settle_worse_abs:
            reasons.append("settle_worse_too_much")
        if teacher_pair_mean > raw_pair_mean + args.max_post_pair_mean_worse_abs:
            reasons.append("post_pair_mean_worse_too_much")
        if metrics["max_delta_t"] > args.max_delta_t:
            reasons.append("delta_t_too_large")
        if metrics["max_delta_r_deg"] > args.max_delta_r_deg:
            reasons.append("delta_r_too_large")

    token_path = output_root / name / "v7_tokens.npz"
    has_tokens = token_path.is_file()
    metrics["has_tokens"] = has_tokens
    if not args.no_require_tokens and not has_tokens:
        reasons.append("missing_tokens")

    return {
        "name": name,
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "tokens_npz": str(token_path),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output_root = output_root_from_manifest(manifest_path, manifest)
    rows = [evaluate_case(case_entry, output_root, args) for case_entry in manifest.get("cases", [])]
    accepted = [row for row in rows if row["status"] == "accepted"]
    rejected = [row for row in rows if row["status"] == "rejected"]
    result = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "num_cases": len(rows),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "accepted_cases": [row["name"] for row in accepted],
        "rejected_cases": {row["name"]: row["reasons"] for row in rejected},
        "thresholds": {
            "min_raw_boundary_foot": args.min_raw_boundary_foot,
            "min_boundary_improve": args.min_boundary_improve,
            "max_boundary_ratio": args.max_boundary_ratio,
            "max_settle_worse_abs": args.max_settle_worse_abs,
            "max_post_pair_mean_worse_abs": args.max_post_pair_mean_worse_abs,
            "max_delta_t": args.max_delta_t,
            "max_delta_r_deg": args.max_delta_r_deg,
            "require_tokens": not args.no_require_tokens,
        },
        "cases": rows,
    }

    output_path = args.output
    if output_path is None:
        output_path = output_root / "stage_a_quality_summary.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
