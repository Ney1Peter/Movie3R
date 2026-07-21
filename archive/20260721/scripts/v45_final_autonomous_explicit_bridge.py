#!/usr/bin/env python3
"""Finalize the autonomous explicit bridge search after H7 validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v36_final_autonomous_explicit_bridge_report import method_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/v45_final_autonomous_explicit_bridge"
DOC = ROOT / "docs/movie3r/v45/V45_FINAL_AUTONOMOUS_EXPLICIT_BRIDGE_20260721.md"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "catastrophic_rate": float(np.mean(array > 45.0)),
    }


def rotation_summary(rows: list[dict]) -> dict:
    output = {
        method: distribution([float(row[f"{method}_error"]) for row in rows])
        for method in ("fixed", "torso", "v32", "v36")
    }
    v32 = np.asarray([row["v32_error"] for row in rows], dtype=np.float64)
    v36 = np.asarray([row["v36_error"] for row in rows], dtype=np.float64)
    output["v36_vs_v32"] = {
        "changed_count": int(np.sum(np.abs(v36 - v32) > 1e-4)),
        "improved_over_5deg_count": int(np.sum(v36 + 5.0 < v32)),
        "harmful_over_5deg_count": int(np.sum(v36 > v32 + 5.0)),
        "rescued_catastrophic_count": int(np.sum((v32 > 45.0) & (v36 <= 45.0))),
        "introduced_catastrophic_count": int(np.sum((v32 <= 45.0) & (v36 > 45.0))),
    }
    return output


def build_rotation_report(h7: dict) -> dict:
    rows, _ = method_rows()
    for row in h7["cases"]:
        rows.append({
            "set": "holdout7_valid",
            "case_name": row["case_name"],
            "source": row["source"],
            "fixed_error": row["fixed_error"],
            "torso_error": row["torso_error"],
            "v32_error": row["v32_error"],
            "v36_error": row["v36_error"],
        })
    set_names = ["original180"] + [f"holdout{i}" for i in range(1, 7)] + ["holdout7_valid"]
    return {
        "valid_case_count": len(rows),
        "holdout7_missing_human_count": 1,
        "combined": rotation_summary(rows),
        "by_set": {
            name: rotation_summary([row for row in rows if row["set"] == name])
            for name in set_names
        },
        "holdout7_frozen_decision": {
            "changed_count": h7["changed_count"],
            "v32_mean_deg": h7["overall"]["v32_rotation_deg"]["mean"],
            "v32_p95_deg": h7["overall"]["v32_rotation_deg"]["p95"],
            "v36_mean_deg": h7["overall"]["v36_rotation_deg"]["mean"],
            "v36_p95_deg": h7["overall"]["v36_rotation_deg"]["p95"],
            "v36_improved_over_5deg": h7["overall"]["v36_improved_over_5deg_vs_v32"],
            "v36_harmful_over_5deg": h7["overall"]["v36_harmful_over_5deg_vs_v32"],
            "decision": "reject V36 adaptive cap; retain V32",
        },
    }


def export_final(v40: dict, v36_end: dict) -> dict:
    scene_by_case = {row["case_name"]: row for row in v36_end["cases"]}
    cases = []
    for row in v40["cases"]:
        selected = row["variants"]["human_root"]
        scene = scene_by_case[row["case_name"]]["scene_scale_sets"]["absolute"]
        cases.append({
            "case_name": row["case_name"],
            "source": row["source"],
            "transform": selected["transform"],
            "human_root_scales": selected["scales"],
            "pointmap_scene_scales": scene,
            "runtime_branch": "V32 rotation + V22 explicit human-root translation",
        })
    return {
        "method": "V45 selected explicit bridge (V32 + V22 scale/translation)",
        "case_count": len(cases),
        "protocol": {
            "post_cut_frames": 1,
            "shot_transform_count": 1,
            "hard_reset": True,
            "runtime_gt": False,
            "camera_pointmap_smplx_share_transform": True,
            "scene_gated_background_scale": "rejected by independent H7",
        },
        "cases": cases,
    }


def markdown(report: dict) -> str:
    rotation = report["rotation_validation"]
    final = report["original180_final_metrics"]
    holdout = report["scale_validation"]["holdout7"]
    lines = [
        "# V45 Final Autonomous Explicit Boundary Bridge",
        "",
        "## Final Decision",
        "",
        "Selected deployable method:",
        "",
        "```text",
        "camera cut",
        "-> hard reset Human3R",
        "-> V22 independent DA3 metric shot scales",
        "-> V16 torso/gravity bounded rotation",
        "-> V32 texture-safe conditional VGGT 1+1 rotation",
        "-> V22 explicit human-root camera translation re-solving",
        "-> one fixed shot-level transform for camera, pointmap, and SMPL-X",
        "```",
        "",
        "V36 human-jump adaptive capping is rejected by H7. The V43 scene-gated background-scale replacement is also rejected by H7.",
        "",
        "## Rotation Validation",
        "",
        "| Set | Count | Fixed mean/P95 | Torso mean/P95 | V32 mean/P95 | V36 mean/P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in rotation["by_set"].items():
        lines.append(
            f"| {name} | {values['v32']['count']} | "
            f"{values['fixed']['mean']:.2f}/{values['fixed']['p95']:.2f} | "
            f"{values['torso']['mean']:.2f}/{values['torso']['p95']:.2f} | "
            f"{values['v32']['mean']:.2f}/{values['v32']['p95']:.2f} | "
            f"{values['v36']['mean']:.2f}/{values['v36']['p95']:.2f} |"
        )
    h7 = rotation["holdout7_frozen_decision"]
    lines.extend([
        "",
        f"Across `{rotation['valid_case_count']}` paired cuts, V32 rotation is the retained rule. On H7, V36 changed `{h7['changed_count']}` cases, improved none by >5 deg, and harmed one by >5 deg (`{h7['v32_mean_deg']:.2f} -> {h7['v36_mean_deg']:.2f} deg` mean).",
        "",
        "## Original 180 End-To-End",
        "",
        "| Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic |",
        "|---:|---:|---:|---:|---:|",
        f"| {final['camera_translation_m']['mean']:.3f}/{final['camera_translation_m']['p95']:.3f} m | "
        f"{final['camera_rotation_deg']['mean']:.2f}/{final['camera_rotation_deg']['p95']:.2f} deg | "
        f"{final['human_motion_error_m']['mean']:.3f} m | "
        f"{final['scene_trimmed_mean_m']['mean']:.3f}/{final['scene_trimmed_mean_m']['p95']:.3f} m | "
        f"{final['combined_catastrophic_rate']:.2%} |",
        "",
        "## Translation Search",
        "",
        f"- V38: 120 acceptable, 15 rotation-dominated, 16 metric-depth-dominated, 7 metric-transverse-dominated, and 21 mixed cases. The residual tail is concentrated in MVHuman.",
        f"- V39: pelvis/torso and one/five-frame root-scale variants differ only at millimeter-level means and do not provide a stable safety gain.",
        f"- V40: unrestricted post-cut background scale improves development mean from `{report['scale_validation']['development']['baseline_mean_m']:.3f}` to `{report['scale_validation']['development']['unrestricted_mean_m']:.3f} m`, but contains harmful corrections.",
        f"- V41/V42: the 2 cm scene gate selected the same five development cases in all nine sampling runs and harmed none there.",
        f"- V44 H7: the frozen gate selected `{holdout['accepted_count']}` cases, improved `{holdout['improved_005m']}`, but harmed `{holdout['harmful_005m']}` by >5 cm and `{holdout['harmful_010m']}` by >10 cm. It is rejected.",
        f"- Post-hoc 7.5% bound: selected `{report['scale_validation']['exploratory_0075_bound']['accepted_count']}` H7 cases and reduced harm to `{report['scale_validation']['exploratory_0075_bound']['harmful_005m']}` >5 cm / `{report['scale_validation']['exploratory_0075_bound']['harmful_010m']}` >10 cm, but did not reach zero harm. It is an H8 hypothesis, not a selected rule.",
        "",
        "## What Failed and Why",
        "",
        "The DA3 background cue can improve metric scale on some MVHuman cuts, but Human3R scene continuity is not a reliable proxy for camera-translation correctness on an unseen set. A candidate can make the two predicted point clouds look closer while moving the real camera farther from GT. Therefore scene-only gating is not a safe deployment rule.",
        "",
        "## Final Scope",
        "",
        "The final method is training-free, cut-only, 1+1 streaming, and uses no token correction, learned selector, learned SE(3), GT depth, full future shot, BA, or recurrent-state edit. One H7 cut has no reset-frame human and must fall back to Fixed Explicit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    h7_rotation = load("output/v37_human_jump_holdout7/frozen_rule_validation_valid/v36_frozen_human_jump_consensus_validation.json")
    v38 = load("output/v38_final_bridge_residual_audit/v38_final_bridge_residual_audit.json")
    v39 = load("output/v39_da3_root_scale_robustness/v39_da3_root_scale_robustness.json")
    v40 = load("output/v45_final_autonomous_explicit_bridge/v40_v32/v40_human_background_scale_fusion.json")
    v42 = load("output/v45_final_autonomous_explicit_bridge/v42_v32_scene_gate_stability_report.json")
    v43 = load("output/v45_final_autonomous_explicit_bridge/v43_v32/v43_scene_gated_metric_bridge.json")
    v44 = load("output/v37_human_jump_holdout7/v44_scale_validation/v44_holdout_scene_gated_scale_validation.json")
    v44_bound = load("output/v37_human_jump_holdout7/v44_scale_bound_0p075/v44_holdout_scene_gated_scale_validation.json")
    v36_end = load("output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json")

    report = {
        "experiment": "V45 final autonomous explicit bridge selection",
        "decision": {
            "selected_method": "V32 rotation + V22 explicit metric translation",
            "v36_adaptive_cap": "rejected by H7",
            "v43_scene_gated_background_scale": "rejected by H7",
            "direct_da3_translation": "rejected because camera and Human3R geometry separate",
        },
        "protocol": {
            "hard_reset": True,
            "post_cut_frames": 1,
            "shot_transform_count": 1,
            "runtime_gt": False,
            "viewer_started": False,
        },
        "rotation_validation": build_rotation_report(h7_rotation),
        "original180_final_metrics": v43["overall"]["v36"],
        "residual_audit": {
            "classification_counts": v38["classification_counts"],
            "diagnostic_counts": v38["diagnostic_counts"],
        },
        "root_scale_robustness": {
            "torso_first1": v39["overall"]["torso_first1"],
            "pelvis_first1": v39["overall"]["pelvis_first1"],
            "torso_median5": v39["overall"]["torso_median5"],
            "pelvis_median5": v39["overall"]["pelvis_median5"],
            "decision": "no robust variant improves enough to replace torso_first1",
        },
        "scale_validation": {
            "development": {
                "scale_disagreement_count": v40["scale_disagreement_count"],
                "baseline_mean_m": v40["overall"]["human_root"]["camera_translation_m"]["mean"],
                "unrestricted_mean_m": v40["overall"]["human_old_background_new"]["camera_translation_m"]["mean"],
                "gated_selected_count": v43["selected_count"],
                "gated_improved_005m": v43["overall"]["v43"]["translation_improved_005m"],
                "gated_harmful_005m": v43["overall"]["v43"]["translation_harmful_005m"],
                "sampling_runs": v42["run_count"],
                "stable_selection_count": v42["intersection_count"],
            },
            "holdout7": {
                "case_count": v44["case_count"],
                "accepted_count": v44["accepted_count"],
                "baseline_mean_m": v44["overall"]["v36"]["camera_translation_m"]["mean"],
                "candidate_mean_m": v44["overall"]["v44"]["camera_translation_m"]["mean"],
                "baseline_p95_m": v44["overall"]["v36"]["camera_translation_m"]["p95"],
                "candidate_p95_m": v44["overall"]["v44"]["camera_translation_m"]["p95"],
                "improved_005m": v44["overall"]["v44"]["translation_improved_005m"],
                "harmful_005m": v44["overall"]["v44"]["translation_harmful_005m"],
                "improved_010m": v44["overall"]["v44"]["translation_improved_010m"],
                "harmful_010m": v44["overall"]["v44"]["translation_harmful_010m"],
                "decision": "reject frozen scene gate",
            },
            "exploratory_0075_bound": {
                "accepted_count": v44_bound["accepted_count"],
                "candidate_mean_m": v44_bound["overall"]["v44"]["camera_translation_m"]["mean"],
                "candidate_p95_m": v44_bound["overall"]["v44"]["camera_translation_m"]["p95"],
                "improved_005m": v44_bound["overall"]["v44"]["translation_improved_005m"],
                "harmful_005m": v44_bound["overall"]["v44"]["translation_harmful_005m"],
                "improved_010m": v44_bound["overall"]["v44"]["translation_improved_010m"],
                "harmful_010m": v44_bound["overall"]["v44"]["translation_harmful_010m"],
                "status": "post-hoc H8 hypothesis only",
            },
        },
    }
    export = export_final(v40, v36_end)
    report_path = OUT / "v45_final_autonomous_explicit_bridge.json"
    export_path = OUT / "v45_selected_explicit_bridge_180.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    export_path.write_text(json.dumps(export, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    DOC.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "rotation_combined": report["rotation_validation"]["combined"],
        "original180": report["original180_final_metrics"],
        "holdout7_scale": report["scale_validation"]["holdout7"],
        "report": str(report_path),
        "export": str(export_path),
        "doc": str(DOC),
    }, indent=2))


if __name__ == "__main__":
    main()
