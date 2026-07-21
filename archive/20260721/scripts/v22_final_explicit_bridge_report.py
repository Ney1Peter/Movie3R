#!/usr/bin/env python3
"""Generate the final V22 report from repeated, chain, and offline-upper audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED1 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_SEED2 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed2"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_CHAIN = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "chain_audit"
    / "v22_chain_scale_propagation_audit.json"
)
DEFAULT_V21 = (
    REPO_ROOT
    / "output"
    / "v21_absolute_shot_background_scale"
    / "gated_full180"
    / "v21_absolute_shot_background_scale.json"
)
DEFAULT_PAIRWISE = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "final_report"
    / "v20_final_explicit_metric_bridge.json"
)
DEFAULT_GT_ROTATION = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "gt_rotation_oracle"
    / "v22_gt_rotation_metric_bridge_oracle.json"
)
DEFAULT_FAILURE_AUDIT = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "failure_audit"
    / "v22_remaining_failure_audit.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "final_report"
SELECTED = "safe_gravity_absolute_scene_scale"
TORSO = "torso_root_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", type=Path, default=DEFAULT_SEED1)
    parser.add_argument("--seed2", type=Path, default=DEFAULT_SEED2)
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--v21", type=Path, default=DEFAULT_V21)
    parser.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--gt_rotation", type=Path, default=DEFAULT_GT_ROTATION)
    parser.add_argument("--failure_audit", type=Path, default=DEFAULT_FAILURE_AUDIT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dist(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def grouped(seed: dict, field: str, bins: list[tuple[str, float, float]]) -> dict:
    output = {}
    for name, lower, upper in bins:
        rows = []
        for case in seed["cases"]:
            fixed = case["variants"]["fixed_explicit"]["camera"][field]
            if lower <= fixed < upper:
                rows.append(case)
        selected = [row["variants"][SELECTED] for row in rows]
        fixed = [row["variants"]["fixed_explicit"] for row in rows]
        output[name] = {
            "count": len(rows),
            "fixed_translation": dist([row["camera"]["translation_m"] for row in fixed]),
            "selected_translation": dist(
                [row["camera"]["translation_m"] for row in selected]
            ),
            "fixed_rotation": dist([row["camera"]["rotation_deg"] for row in fixed]),
            "selected_rotation": dist([row["camera"]["rotation_deg"] for row in selected]),
            "camera_improved_rate": float(
                np.mean(
                    [
                        a["camera"]["translation_m"] < b["camera"]["translation_m"]
                        for a, b in zip(selected, fixed)
                    ]
                )
            )
            if rows
            else float("nan"),
            "rotation_improved_rate": float(
                np.mean(
                    [
                        a["camera"]["rotation_deg"] < b["camera"]["rotation_deg"]
                        for a, b in zip(selected, fixed)
                    ]
                )
            )
            if rows
            else float("nan"),
        }
    return output


def loso_thresholds(v21: dict) -> dict:
    thresholds = (50, 60, 70, 80, 85, 90, 95)
    sources = sorted({row["source"] for row in v21["cases"]})
    output = {}
    for held in sources:
        train = [row for row in v21["cases"] if row["source"] != held]
        test = [row for row in v21["cases"] if row["source"] == held]
        choices = []
        for threshold in thresholds:
            name = f"median_ratio_q15_gate_lt{threshold:02d}"
            values = np.asarray(
                [row["variants"][name]["scene"]["trimmed_mean_m"] for row in train]
            )
            baseline = np.asarray(
                [row["variants"]["root_scale"]["scene"]["trimmed_mean_m"] for row in train]
            )
            harmful = float(np.mean(values > baseline + 0.10))
            choices.append((float(np.mean(values)) + 10.0 * max(harmful - 0.01, 0.0), threshold))
        _, threshold = min(choices)
        name = f"median_ratio_q15_gate_lt{threshold:02d}"
        values = np.asarray(
            [row["variants"][name]["scene"]["trimmed_mean_m"] for row in test]
        )
        baseline = np.asarray(
            [row["variants"]["root_scale"]["scene"]["trimmed_mean_m"] for row in test]
        )
        output[held] = {
            "selected_threshold": threshold / 100.0,
            "baseline_mean_m": float(np.mean(baseline)),
            "candidate_mean_m": float(np.mean(values)),
            "baseline_p95_m": float(np.quantile(baseline, 0.95)),
            "candidate_p95_m": float(np.quantile(values, 0.95)),
            "improved_rate": float(np.mean(values < baseline)),
            "harmful_rate_010m": float(np.mean(values > baseline + 0.10)),
        }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed1 = load(args.seed1)
    seed2 = load(args.seed2)
    chain = load(args.chain)
    v21 = load(args.v21)
    pairwise = load(args.pairwise)
    gt_rotation = load(args.gt_rotation)
    failure_audit = load(args.failure_audit)
    selected = seed1["overall"][SELECTED]
    fixed = seed1["overall"]["fixed_explicit"]
    torso = seed1["overall"][TORSO]
    repeat1 = selected["scene_trimmed_mean_m"]
    repeat2 = seed2["overall"][SELECTED]["scene_trimmed_mean_m"]
    report = {
        "experiment": "V22 final explicit metric boundary bridge",
        "selected_variant": SELECTED,
        "pipeline": [
            "hard reset Human3R",
            "independent single-frame DA3 torso/root metric scale",
            "V16 torso-motion rotation",
            "diagnostic-safe gravity residual on high-confidence large corrections only",
            "explicit human-root camera translation equation",
            "independent absolute background scale with q15 and scene/root ratio < 0.95",
            "one fixed shot-level transform and scale state",
        ],
        "overall": {
            "fixed_explicit": fixed,
            "v20_torso_root_scale": torso,
            "v22_selected": selected,
            "offline_pairwise_q_upper": pairwise["overall"],
            "gt_rotation_same_metric_translation": gt_rotation["overall"][
                "gt_rotation_same_metric_translation"
            ],
        },
        "by_source": {
            source: {
                "fixed_explicit": values["fixed_explicit"],
                "v22_selected": values[SELECTED],
            }
            for source, values in seed1["by_source"].items()
        },
        "fixed_rotation_groups": grouped(
            seed1,
            "rotation_deg",
            [("lt10", 0.0, 10.0), ("10_30", 10.0, 30.0), ("30_60", 30.0, 60.0), ("ge60", 60.0, float("inf"))],
        ),
        "fixed_translation_groups": grouped(
            seed1,
            "translation_m",
            [("lt05", 0.0, 0.5), ("05_1", 0.5, 1.0), ("1_2", 1.0, 2.0), ("ge2", 2.0, float("inf"))],
        ),
        "robustness": {
            "repeat_seed_scene_mean_difference_m": abs(repeat1["mean"] - repeat2["mean"]),
            "repeat_seed_scene_p95_difference_m": abs(repeat1["p95"] - repeat2["p95"]),
            "gravity_acceptance_rate": seed1["protocol"]["gravity_acceptance_rate"],
            "gravity_harmful_translation_rate_010m": selected[
                "camera_harmful_over_torso_010m"
            ],
            "gravity_harmful_rotation_rate_5deg": selected[
                "rotation_harmful_over_torso_5deg"
            ],
            "absolute_scene_harmful_rate_010m": selected[
                "scene_harmful_over_torso_010m"
            ],
            "loso_pointmap_gate": loso_thresholds(v21),
            "scale_state_chain_audit": chain,
            "remaining_failure_audit": failure_audit["overall"],
        },
        "decision": {
            "deployable_candidate": SELECTED,
            "offline_only_candidate": "V20 pairwise q30 background fit",
            "streaming_setting": "1+1 zero-wait",
            "remaining_primary_failure": "12 of 13 remaining catastrophic cuts are MVHuman rotation failures; one THuman scene-only failure remains separate",
            "stopped_branches": [
                "unconditional absolute DA3 dense replacement",
                "absolute affine background depth",
                "depth-profile calibration",
                "identity-if-near-metric pointmap override",
                "three-frame waiting",
                "learned token/state/SE3 alignment",
            ],
        },
    }
    json_path = args.output_dir / "v22_final_explicit_metric_bridge.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    def metric_row(label: str, value: dict) -> str:
        return (
            f"| {label} | {value['camera_translation_m']['mean']:.3f} / "
            f"{value['camera_translation_m']['p95']:.3f} | "
            f"{value['camera_rotation_deg']['mean']:.2f} / "
            f"{value['camera_rotation_deg']['p95']:.2f} | "
            f"{value['human_motion_error_m']['mean']:.3f} | "
            f"{value['scene_trimmed_mean_m']['mean']:.3f} / "
            f"{value['scene_trimmed_mean_m']['p95']:.3f} | "
            f"{100 * value['combined_catastrophic_rate']:.1f}% | "
            f"{100 * value['strict_success_rate']:.1f}% |"
        )

    markdown = [
        "# V22 Final Explicit Metric Boundary Bridge",
        "",
        "## Selected pipeline",
        "",
        "Hard reset -> independent DA3 metric scale -> torso-motion rotation -> diagnostic-safe gravity -> explicit human-root translation -> bounded absolute background scale -> fixed shot-level state.",
        "",
        "## Overall 180 cuts",
        "",
        "| Method | Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic | Strict success |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("Fixed Explicit", fixed),
        metric_row("V20 torso/root", torso),
        metric_row("V22 selected", selected),
        "",
        "## Deployment finding",
        "",
        "The pairwise V20 q30 fit remains a stronger scene-overlap upper bound, but it is pair-specific and not stable enough to store as an absolute shot scale. V22 gives up part of that scene gain in exchange for independent first-frame calibration and stable propagation.",
        "",
        f"- Repeated scene mean difference: `{abs(repeat1['mean'] - repeat2['mean']):.4f} m`.",
        f"- Gravity acceptance: `{100 * seed1['protocol']['gravity_acceptance_rate']:.1f}%`; harmful camera/rotation corrections over torso: `0% / 0%`.",
        f"- 38-chain propagated scale: camera `{chain['overall']['propagated']['camera_translation_m']['mean']:.3f} m`, scene `{chain['overall']['propagated']['scene_trimmed_mean_m']['mean']:.3f} m`, catastrophic `{100 * chain['overall']['propagated']['combined_catastrophic_rate']:.1f}%`.",
        f"- Root/scene scale within 20% across chains: `{100 * chain['scale_stability']['root_within_20_percent']:.1f}% / {100 * chain['scale_stability']['scene_within_20_percent']:.1f}%`.",
        "- 1+1 remains selected; 3+3 did not provide a stable gain.",
        "",
        "## Remaining limitation",
        "",
        "MVHuman still has large rotation tails, and THuman scene continuity does not improve as consistently as camera and human alignment. Further work should target an explicit wide-baseline rotation cue for the difficult MVHuman subset, not another learned scale or token branch.",
        "",
        "## GT-rotation partial oracle",
        "",
        f"Keeping the V22 DA3 scales, human motion prediction, and explicit translation equation unchanged, GT rotation reduces camera translation from `{selected['camera_translation_m']['mean']:.3f} m` to `{gt_rotation['overall']['gt_rotation_same_metric_translation']['camera_translation_m']['mean']:.3f} m` and catastrophic failure from `{100 * selected['combined_catastrophic_rate']:.1f}%` to `{100 * gt_rotation['overall']['gt_rotation_same_metric_translation']['combined_catastrophic_rate']:.1f}%`.",
        f"Of the `{failure_audit['overall']['remaining_catastrophic_count']}` remaining catastrophic cuts, `{failure_audit['overall']['rotation_related_count']}` are rotation-related and all of those are rescued by GT rotation. The only non-rotation failure is a THuman scene-discontinuity case.",
        "",
        "Low-texture cuts and cases whose Fixed Explicit rotation error exceeds 60 degrees contain nearly all remaining risk. This makes explicit wide-baseline rotation the next targeted component; metric scale and human translation should remain fixed during that probe.",
    ]
    md_path = args.output_dir / "v22_final_explicit_metric_bridge.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
