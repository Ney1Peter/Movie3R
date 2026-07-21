#!/usr/bin/env python3
"""Audit the deployable low-torso VGGT pretrigger used by the V25 bridge."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = (
    REPO_ROOT
    / "output"
    / "v25_explicit_consensus_bridge"
    / "emulation_audit"
    / "v25_holdout_rotation_validation.json"
)
DEFAULT_V15 = (
    REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v25_explicit_consensus_bridge" / "runtime_pretrigger_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--texture_thresholds",
        type=float,
        nargs="+",
        default=(0.020, 0.025, 0.030, 0.050),
    )
    parser.add_argument("--untriggered_latency", type=float, default=0.3515626907)
    parser.add_argument("--triggered_latency", type=float, default=0.8611574017)
    return parser.parse_args()


def load_v15(root: Path) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {row["case_name"]: row for row in rows}


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def rotation_metrics(errors: np.ndarray, baseline: np.ndarray) -> dict:
    return {
        "rotation_deg": distribution(errors),
        "catastrophic_rate": float(np.mean(errors > 45.0)),
        "rescued_catastrophic_count": int(
            np.sum((baseline > 45.0) & (errors <= 45.0))
        ),
        "introduced_catastrophic_count": int(
            np.sum((baseline <= 45.0) & (errors > 45.0))
        ),
        "harmful_over_5deg_count": int(np.sum(errors > baseline + 5.0)),
        "good_case_harmful_count": int(
            np.sum((baseline < 10.0) & (errors > baseline + 5.0))
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    v15 = load_v15(args.v15_dir)
    rows = evaluation["cases"]
    if not rows or set(v15) != {row["case_name"] for row in rows}:
        raise RuntimeError(
            f"Evaluation/V15 case mismatch: {len(rows)} evaluation, {len(v15)} V15"
        )

    baseline = np.asarray([row["v24"]["rotation_error_deg"] for row in rows])
    full_v25 = np.asarray([row["v25"]["rotation_error_deg"] for row in rows])
    current_pretrigger = np.asarray(
        [row["v24_diagnostics"]["torso_residual_deg"] >= 10.0 for row in rows],
        dtype=bool,
    )
    explicit_trigger = np.asarray(
        [
            row["v25_diagnostics"]["trigger_low_torso_explicit_consensus"]
            for row in rows
        ],
        dtype=bool,
    )
    background_trigger = np.asarray(
        [row["v25_diagnostics"]["trigger_background_1p1_fallback"] for row in rows],
        dtype=bool,
    )
    v24_accepted = np.asarray(
        [row["v24_diagnostics"]["v24_accepted"] for row in rows], dtype=bool
    )
    torso_residual = np.asarray(
        [row["v24_diagnostics"]["torso_residual_deg"] for row in rows]
    )
    textures = np.asarray([float(v15[row["case_name"]]["texture_score"]) for row in rows])
    full_spread = np.asarray(
        [
            float(
                v15[row["case_name"]]["windows"]["full_rgb_1p1"][
                    "rotation_consensus"
                ]["spread_deg"]
            )
            for row in rows
        ]
    )
    sources = np.asarray([row["source"] for row in rows])
    background_second_stage = (
        (~v24_accepted) & (torso_residual >= 30.0) & (full_spread > 15.0)
    )
    vggt_incremental_latency = args.triggered_latency - args.untriggered_latency

    audits = []
    for threshold in args.texture_thresholds:
        additional = (~current_pretrigger) & (textures < threshold)
        available_explicit = explicit_trigger & additional
        deployable_errors = np.where(available_explicit | background_trigger, full_v25, baseline)
        total_pretrigger = current_pretrigger | additional
        trigger_rate = float(np.mean(total_pretrigger))
        mean_latency = (
            trigger_rate * args.triggered_latency
            + (1.0 - trigger_rate) * args.untriggered_latency
            + float(np.mean(background_second_stage)) * vggt_incremental_latency
        )
        current_rate = float(np.mean(current_pretrigger))
        current_latency = (
            current_rate * args.triggered_latency
            + (1.0 - current_rate) * args.untriggered_latency
        )
        source_rows = {}
        for source in sorted(set(sources.tolist())):
            mask = sources == source
            source_rows[source] = {
                "case_count": int(mask.sum()),
                "additional_pretrigger_count": int(np.sum(additional & mask)),
                "available_explicit_consensus_count": int(
                    np.sum(available_explicit & mask)
                ),
                "rotation": rotation_metrics(
                    deployable_errors[mask], baseline[mask]
                ),
            }
        audits.append(
            {
                "texture_threshold": float(threshold),
                "current_v24_pretrigger_count": int(current_pretrigger.sum()),
                "additional_low_torso_pretrigger_count": int(additional.sum()),
                "total_pretrigger_count": int(total_pretrigger.sum()),
                "total_pretrigger_rate": trigger_rate,
                "background_second_stage_probe_count": int(
                    background_second_stage.sum()
                ),
                "background_fallback_count": int(background_trigger.sum()),
                "background_probe_precision": float(
                    background_trigger.sum()
                    / max(int(background_second_stage.sum()), 1)
                ),
                "available_explicit_consensus_count": int(available_explicit.sum()),
                "useful_pretrigger_precision": float(
                    available_explicit.sum() / max(int(additional.sum()), 1)
                ),
                "missed_explicit_consensus_count": int(
                    np.sum(explicit_trigger & ~additional)
                ),
                "estimated_mean_cut_latency_seconds": float(mean_latency),
                "estimated_latency_increase_over_v24_seconds": float(
                    mean_latency - current_latency
                ),
                "rotation": rotation_metrics(deployable_errors, baseline),
                "by_source": source_rows,
            }
        )

    deployable = [
        row
        for row in audits
        if row["missed_explicit_consensus_count"] == 0
        and row["rotation"]["introduced_catastrophic_count"] == 0
        and row["rotation"]["harmful_over_5deg_count"] == 0
    ]
    selected = min(
        deployable or audits,
        key=lambda row: (
            row["missed_explicit_consensus_count"],
            row["additional_low_torso_pretrigger_count"],
            row["rotation"]["rotation_deg"]["mean"],
        ),
    )
    report = {
        "experiment": "V25 low-torso runtime VGGT pretrigger audit",
        "case_count": len(rows),
        "protocol": {
            "pre_v25_vggt_trigger": "torso residual >= 10 deg",
            "new_cheap_pretrigger": "torso residual < 10 deg and texture < threshold",
            "background_second_stage_precondition": (
                "V24 rejected, torso residual >= 30 deg, full-RGB spread > 15 deg"
            ),
            "texture_available_before_vggt": True,
            "latency_source": "V24 integrated resident-model benchmark",
            "gt_runtime_information": False,
        },
        "full_offline_v25_rotation": rotation_metrics(full_v25, baseline),
        "threshold_audits": audits,
        "selected_threshold": selected["texture_threshold"],
        "selected": selected,
    }
    output = args.output_dir / "v25_low_torso_runtime_pretrigger_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "selected_threshold": report["selected_threshold"],
                "threshold_audits": audits,
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
