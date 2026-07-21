#!/usr/bin/env python3
"""Build the final V36 autonomous explicit metric-bridge report."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output/v36_final_autonomous_explicit_bridge"
DOC = REPO_ROOT / "docs/movie3r/v36/V36_AUTONOMOUS_EXPLICIT_METRIC_BRIDGE_20260721.md"
TEXTURE_BOUND = 0.05
HUMAN_JUMP_BOUND_DEG = 30.0
LOW_JUMP_CONSENSUS_CAP_DEG = 20.0
SETS = (
    (
        "original180",
        REPO_ROOT / "output/v15_wide_baseline_boundary_bridge/candidate_cache",
        REPO_ROOT / "output/v16_human_aware_rotation_residual/candidate_cache",
    ),
    (
        "holdout1",
        REPO_ROOT / "output/v25_holdout_rotation_validation/v15",
        REPO_ROOT / "output/v25_holdout_rotation_validation/v16",
    ),
    (
        "holdout2",
        REPO_ROOT / "output/v27_consensus_holdout2/v15",
        REPO_ROOT / "output/v27_consensus_holdout2/v16",
    ),
    (
        "holdout3",
        REPO_ROOT / "output/v29_rotation_rule_holdout3/v15",
        REPO_ROOT / "output/v29_rotation_rule_holdout3/v16",
    ),
    (
        "holdout4",
        REPO_ROOT / "output/v31_metric_fit_holdout4/v15",
        REPO_ROOT / "output/v31_metric_fit_holdout4/v16",
    ),
    (
        "holdout5",
        REPO_ROOT / "output/v32_texture_safe_holdout5/v15",
        REPO_ROOT / "output/v32_texture_safe_holdout5/v16",
    ),
    (
        "holdout6",
        REPO_ROOT / "output/v36_human_jump_holdout6/v15",
        REPO_ROOT / "output/v36_human_jump_holdout6/v16",
    ),
)


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid cache {root}: {len(rows)}/{len(output)}")
    return output


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "catastrophic_rate": float(np.mean(values > 45.0)),
    }


def aggregate(rows: list[dict]) -> dict:
    arrays = {
        method: np.asarray([row[f"{method}_error"] for row in rows])
        for method in ("fixed", "torso", "v24", "v32", "v36")
    }
    output = {method: distribution(values) for method, values in arrays.items()}
    for baseline in ("fixed", "torso", "v24", "v32"):
        base = arrays[baseline]
        final = arrays["v36"]
        output[f"v36_vs_{baseline}"] = {
            "rescued_catastrophic_count": int(
                np.sum((base > 45.0) & (final <= 45.0))
            ),
            "introduced_catastrophic_count": int(
                np.sum((base <= 45.0) & (final > 45.0))
            ),
            "harmful_over_5deg_count": int(np.sum(final > base + 5.0)),
            "improved_over_5deg_count": int(np.sum(final + 5.0 < base)),
        }
    return output


def method_rows() -> tuple[list[dict], list[dict]]:
    rows = []
    adapted = []
    for set_name, v15_dir, v16_dir in SETS:
        v15 = load_shards(v15_dir, "v15_candidates")
        v16 = load_shards(v16_dir, "v16_candidates")
        names = sorted(set(v15) & set(v16))
        if len(names) != len(v15) or len(names) != len(v16):
            raise RuntimeError(f"Mismatch in {set_name}: {len(v15)}/{len(v16)}")
        for name in names:
            wide = v15[name]
            human = v16[name]
            fixed = np.asarray(
                wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
            )[:3, :3]
            gt = np.asarray(
                wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32
            )[:3, :3]
            torso, gravity = safe_gravity(human)
            candidate_key = (
                "fixed_torso_motion_gravity_1f_resolve_t"
                if gravity["accepted"]
                else "fixed_torso_motion_1f_resolve_t"
            )
            human_jump = float(
                human["fixed_candidates"][candidate_key]["human_torso_jump_deg"]
            )
            v24, _, _ = selected_rotation(
                fixed, torso, wide, None, consensus_cap_deg=60.0
            )
            v32, branch, diagnostics = selected_rotation(
                fixed, torso, wide, TEXTURE_BOUND, consensus_cap_deg=60.0
            )
            use_adaptive_cap = bool(
                branch == "consensus" and human_jump < HUMAN_JUMP_BOUND_DEG
            )
            v36 = (
                selected_rotation(
                    fixed,
                    torso,
                    wide,
                    TEXTURE_BOUND,
                    consensus_cap_deg=LOW_JUMP_CONSENSUS_CAP_DEG,
                )[0]
                if use_adaptive_cap
                else v32
            )
            row = {
                "set": set_name,
                "case_name": name,
                "source": human["record"]["source"],
                "fixed_error": angle_deg(fixed, gt),
                "torso_error": angle_deg(torso, gt),
                "v24_error": angle_deg(v24, gt),
                "v32_error": angle_deg(v32, gt),
                "v36_error": angle_deg(v36, gt),
                "v32_branch": branch,
                "human_torso_jump_deg": human_jump,
                "adaptive_cap_used": use_adaptive_cap,
            }
            rows.append(row)
            if use_adaptive_cap and angle_deg(v32, v36) > 1e-4:
                adapted.append({**row, "diagnostics": diagnostics})
    return rows, adapted


def markdown(report: dict) -> str:
    combined = report["rotation_validation"]["combined"]
    end = report["end_to_end_original180"]["overall_vs_v22"]
    joint = report["direct_da3_joint_audit"]
    selected = report["decision"]["selected_method"]
    lines = [
        "# V36 Autonomous Explicit Metric Boundary Bridge",
        "",
        "## Final Decision",
        "",
        f"Selected method: **{selected}**.",
        "",
        "```text",
        "camera cut",
        "-> hard reset Human3R",
        "-> independent DA3 root/background metric shot scales",
        "-> V16 torso/gravity bounded rotation",
        "-> texture-safe conditional VGGT full-RGB 1+1 rotation",
        "-> if post-torso human heading jump < 30 deg, cap positive-consensus residual at 20 deg",
        "-> explicit metric human-root camera translation re-solving",
        "-> one fixed shot-level scale state and one final SE(3)",
        "-> apply the same final SE(3) to camera, pointmap, and SMPL-X",
        "```",
        "",
        "No token correction, learned gate/selector, learned SE(3), recurrent-state edit, full-shot future, BA, GT depth, or runtime GT is used.",
        "",
        "## Rotation Validation",
        "",
        "| Set | Count | Fixed mean/P95 | Torso mean/P95 | V32 mean/P95 | V36 mean/P95 | V36 cat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["rotation_validation"]["by_set"].items():
        lines.append(
            f"| {name} | {values['count']} | "
            f"{values['fixed']['mean']:.2f}/{values['fixed']['p95']:.2f} | "
            f"{values['torso']['mean']:.2f}/{values['torso']['p95']:.2f} | "
            f"{values['v32']['mean']:.2f}/{values['v32']['p95']:.2f} | "
            f"{values['v36']['mean']:.2f}/{values['v36']['p95']:.2f} | "
            f"{values['v36']['catastrophic_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Across `{report['rotation_validation']['case_count']}` disjoint cuts, V36 rotation mean/P95 is `{combined['v36']['mean']:.2f}/{combined['v36']['p95']:.2f} deg` with `{combined['v36']['catastrophic_rate']:.2%}` catastrophic rate.",
            f"Relative to torso, it rescues `{combined['v36_vs_torso']['rescued_catastrophic_count']}` and introduces `{combined['v36_vs_torso']['introduced_catastrophic_count']}` catastrophes.",
            f"The frozen H6 check changed `{report['frozen_holdout6']['adapted_count']}` cases and introduced `{report['frozen_holdout6']['overall']['v36_introduced_vs_v32']}` catastrophes versus V32.",
            "",
            "## End-To-End 3D Output",
            "",
            "| Method | Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("v22", "v32", "v36", "gt_rotation"):
        value = end[method]
        lines.append(
            f"| {method} | {value['camera_translation_m']['mean']:.3f}/{value['camera_translation_m']['p95']:.3f} m | "
            f"{value['camera_rotation_deg']['mean']:.2f}/{value['camera_rotation_deg']['p95']:.2f} deg | "
            f"{value['human_motion_error_m']['mean']:.3f} m | "
            f"{value['scene_trimmed_mean_m']['mean']:.3f}/{value['scene_trimmed_mean_m']['p95']:.3f} m | "
            f"{value['combined_catastrophic_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The synchronized metric bridge preserves the human result while rotation changes: V36 versus V32 has no >0.1 m harmful human or scene correction on the original 180 cuts.",
            "",
            "## DA3 Role",
            "",
            f"Direct DA3 fixed-rotation SE(3) translation reaches `{joint['da3_camera_mean_m']:.3f} m` camera error but produces `{joint['da3_root_jump_mean_m']:.3f} m` visible Human3R root jump. `{joint['camera_gain_root_harm_rate']:.1%}` of trusted-rotation cases improve camera by >0.1 m while harming root continuity by >0.1 m.",
            "",
            "Therefore DA3 is retained only as an independent metric shot-scale cue inside V22/V36. Direct DA3 boundary translation is rejected.",
            "",
            "## Rejected Extensions",
            "",
            "- Unconditional VGGT rotation: strong tail regressions, especially AvatarReX.",
            "- V25 background fallback: introduced catastrophes on unseen holdout.",
            "- V29/V30 multi-window fallback: no repeatable unseen trigger.",
            "- V31 metric-fit fallback: isolated rescue only, no stable cross-holdout activation.",
            "- V35 scene-metric veto: can reject the only good VGGT rescue when Human3R geometry itself is inconsistent.",
            "- Direct DA3 SE(3) translation: camera improves but Human3R human/scene geometry separates.",
            "",
            "## Remaining Limitation",
            "",
            "The remaining tail is concentrated in MVHuman large-view changes where torso, VGGT full RGB, background, and multi-frame estimates disagree. No additional fixed rule passed independent validation, so those cases remain unresolved instead of adding an unsafe fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    rows, adapted = method_rows()
    by_set = {
        set_name: {
            "count": sum(row["set"] == set_name for row in rows),
            **aggregate([row for row in rows if row["set"] == set_name]),
        }
        for set_name, _, _ in SETS
    }
    frozen = json.loads(
        (
            REPO_ROOT
            / "output/v36_human_jump_holdout6/frozen_rule_validation/v36_frozen_human_jump_consensus_validation.json"
        ).read_text(encoding="utf-8")
    )
    end_to_end = json.loads(
        (
            REPO_ROOT
            / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
        ).read_text(encoding="utf-8")
    )
    joint = json.loads(
        (
            REPO_ROOT
            / "output/v33_joint_camera_human_translation/v33_joint_camera_human_translation_audit.json"
        ).read_text(encoding="utf-8")
    )["trusted_rotation"]
    da3 = joint["da3_vggt_rotation_translation"]
    h6 = frozen["overall"]
    h6_safe = bool(
        h6["v36_catastrophic_rate"] <= h6["v32_catastrophic_rate"]
        and h6["v36_introduced_vs_v32"] == 0
        and h6["v36_harmful_over_5deg_vs_v32"] == 0
    )
    report = {
        "experiment": "V36 final autonomous explicit metric boundary bridge",
        "decision": {
            "selected_method": (
                "V36 human-jump-adaptive explicit metric bridge"
                if h6_safe
                else "V32 texture-safe explicit metric bridge"
            ),
            "v36_passed_frozen_holdout6": h6_safe,
            "direct_da3_translation_rejected": True,
        },
        "protocol": {
            "texture_bound": TEXTURE_BOUND,
            "human_jump_bound_deg": HUMAN_JUMP_BOUND_DEG,
            "low_jump_consensus_cap_deg": LOW_JUMP_CONSENSUS_CAP_DEG,
            "gt_runtime_information": False,
            "viewer_started": False,
        },
        "rotation_validation": {
            "case_count": len(rows),
            "combined": aggregate(rows),
            "by_set": by_set,
            "by_source": {
                source: aggregate([row for row in rows if row["source"] == source])
                for source in sorted({row["source"] for row in rows})
            },
            "adaptive_cap_changed_count": len(adapted),
            "adaptive_cap_changed_cases": adapted,
        },
        "frozen_holdout6": {
            "adapted_count": frozen["adapted_count"],
            "overall": frozen["overall"],
            "by_source": frozen["by_source"],
        },
        "end_to_end_original180": {
            "adapted_count": end_to_end["adapted_count"],
            "overall_vs_v22": end_to_end["overall_vs_v22"],
            "overall_vs_v32": end_to_end["overall_vs_v32"],
        },
        "direct_da3_joint_audit": {
            "trusted_rotation_case_count": da3["camera_translation_m"]["count"],
            "da3_camera_mean_m": da3["camera_translation_m"]["mean"],
            "da3_root_jump_mean_m": da3["visible_human_root_jump_m"]["mean"],
            "camera_gain_root_harm_rate": da3[
                "camera_gain_but_root_harmed_010m_rate"
            ],
            "both_improved_rate": da3["both_improved_rate"],
        },
    }
    output = OUTPUT / "v36_final_autonomous_explicit_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    DOC.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "rotation_case_count": report["rotation_validation"]["case_count"],
                "rotation_combined": report["rotation_validation"]["combined"],
                "frozen_holdout6": report["frozen_holdout6"],
                "direct_da3_joint_audit": report["direct_da3_joint_audit"],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")
    print(f">> wrote {DOC}")


if __name__ == "__main__":
    main()
