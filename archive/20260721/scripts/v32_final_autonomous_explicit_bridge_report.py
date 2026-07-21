#!/usr/bin/env python3
"""Build the final report for the autonomous explicit boundary-bridge search."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output" / "v32_final_autonomous_explicit_bridge"
DOC = REPO_ROOT / "docs" / "movie3r" / "v32" / "V32_AUTONOMOUS_EXPLICIT_BRIDGE_20260721.md"
TEXTURE_BOUND = 0.05
SETS = (
    (
        "original180",
        REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache",
        REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache",
    ),
    (
        "holdout1",
        REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v15",
        REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v16",
    ),
    (
        "holdout2",
        REPO_ROOT / "output" / "v27_consensus_holdout2" / "v15",
        REPO_ROOT / "output" / "v27_consensus_holdout2" / "v16",
    ),
    (
        "holdout3",
        REPO_ROOT / "output" / "v29_rotation_rule_holdout3" / "v15",
        REPO_ROOT / "output" / "v29_rotation_rule_holdout3" / "v16",
    ),
    (
        "holdout4",
        REPO_ROOT / "output" / "v31_metric_fit_holdout4" / "v15",
        REPO_ROOT / "output" / "v31_metric_fit_holdout4" / "v16",
    ),
    (
        "holdout5",
        REPO_ROOT / "output" / "v32_texture_safe_holdout5" / "v15",
        REPO_ROOT / "output" / "v32_texture_safe_holdout5" / "v16",
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
        for method in ("fixed", "torso", "v24", "v32")
    }
    output = {method: distribution(values) for method, values in arrays.items()}
    for baseline in ("fixed", "torso", "v24"):
        base = arrays[baseline]
        final = arrays["v32"]
        output[f"v32_vs_{baseline}"] = {
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


def markdown_table(report: dict) -> str:
    lines = [
        "# V32 Autonomous Explicit Boundary Bridge",
        "",
        "## Decision",
        "",
        "The retained route is an explicit, cut-only cascade:",
        "",
        "```text",
        "hard reset Human3R",
        "-> Fixed Explicit metric gauge",
        "-> bounded torso/gravity rotation",
        "-> texture-safe conditional VGGT 1+1 rotation",
        "-> explicit metric translation re-solving",
        "-> one fixed SE(3) applied to camera, pointmap, and SMPL-X",
        "```",
        "",
        "The positive-consensus VGGT branch is now limited to texture score <= 0.05. Large-residual and low-texture-conflict branches are unchanged.",
        "",
        "## Rotation Validation",
        "",
        "| Set | Count | Fixed mean/P95 | Torso mean/P95 | V24 mean/P95 | V32 mean/P95 | V32 cat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for set_name, values in report["by_set"].items():
        lines.append(
            "| {name} | {count} | {fm:.2f}/{fp:.2f} | {tm:.2f}/{tp:.2f} | "
            "{vm:.2f}/{vp:.2f} | {sm:.2f}/{sp:.2f} | {cat:.2%} |".format(
                name=set_name,
                count=values["count"],
                fm=values["fixed"]["mean"],
                fp=values["fixed"]["p95"],
                tm=values["torso"]["mean"],
                tp=values["torso"]["p95"],
                vm=values["v24"]["mean"],
                vp=values["v24"]["p95"],
                sm=values["v32"]["mean"],
                sp=values["v32"]["p95"],
                cat=values["v32"]["catastrophic_rate"],
            )
        )
    combined = report["combined"]
    lines.extend(
        [
            "",
            "## Combined Safety",
            "",
            f"- Total cuts: `{report['case_count']}`.",
            f"- V32 rotation mean/P95: `{combined['v32']['mean']:.2f}/{combined['v32']['p95']:.2f} deg`.",
            f"- V32 catastrophic rate: `{combined['v32']['catastrophic_rate']:.2%}`.",
            f"- V32 vs torso: rescued `{combined['v32_vs_torso']['rescued_catastrophic_count']}`, introduced `{combined['v32_vs_torso']['introduced_catastrophic_count']}` catastrophes.",
            f"- V32 changed `{report['texture_guard_changed_count']}` V24 decisions.",
            "",
            "## Metric Translation",
            "",
            "With VGGT rotation error <= 45 deg, DA3 fixed-rotation translation is consistently better than Human3R pointmap translation:",
            "",
            "| Set | DA3 mean/P95 | Human3R mean/P95 |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["da3_translation"].items():
        lines.append(
            f"| {name} | {values['da3_mean']:.3f}/{values['da3_p95']:.3f} m | "
            f"{values['human3r_mean']:.3f}/{values['human3r_p95']:.3f} m |"
        )
    lines.extend(
        [
            "",
            "DA3 must not estimate free rotation. Its role is metric translation after a trusted rotation is available.",
            "",
            "## Rejected Extensions",
            "",
            "- V25 background and low-torso fallbacks are not retained; H4 introduced two catastrophes.",
            "- V29/V30 multi-window rules had no trigger on the next unseen holdout.",
            "- V31 low-torso metric-fit fallback had no trigger on H4 and remains diagnostic only.",
            "- Essential rotation and free DA3 3D-3D rotation are not reliable enough for unconditional use.",
            "",
            "## Remaining Work",
            "",
            "The next implementation task is to connect the selected V32 rotation to DA3 fixed-rotation translation re-solving in the runtime path, then export one unified transform for camera, pointmap, and SMPL-X. V22 remains the deployable translation path until that integration receives end-to-end scene and human continuity validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    changed = []
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
            torso, _ = safe_gravity(human)
            v24, _, _ = selected_rotation(fixed, torso, wide, None)
            v32, branch, diagnostics = selected_rotation(
                fixed, torso, wide, TEXTURE_BOUND
            )
            row = {
                "set": set_name,
                "case_name": name,
                "source": human["record"]["source"],
                "fixed_error": angle_deg(fixed, gt),
                "torso_error": angle_deg(torso, gt),
                "v24_error": angle_deg(v24, gt),
                "v32_error": angle_deg(v32, gt),
                "v32_branch": branch,
            }
            rows.append(row)
            if angle_deg(v24, v32) > 1e-4:
                changed.append({**row, "diagnostics": diagnostics})

    da3_translation = {}
    for name, path in (
        (
            "holdout3",
            REPO_ROOT
            / "output"
            / "v29_rotation_rule_holdout3"
            / "da3_translation_audit"
            / "v30_da3_translation_stability_audit.json",
        ),
        (
            "holdout4",
            REPO_ROOT
            / "output"
            / "v31_metric_fit_holdout4"
            / "da3_translation_audit"
            / "v30_da3_translation_stability_audit.json",
        ),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))["by_window"]["full_rgb_1p1"][
            "rotation_le_45deg"
        ]
        da3 = data["da3_fixed_rotation_translation_m"]
        human3r = data["human3r_fixed_rotation_translation_m"]
        da3_translation[name] = {
            "da3_mean": da3["mean"],
            "da3_p95": da3["p95"],
            "human3r_mean": human3r["mean"],
            "human3r_p95": human3r["p95"],
        }

    report = {
        "experiment": "V32 final autonomous explicit bridge report",
        "case_count": len(rows),
        "protocol": {
            "texture_safe_consensus_bound": TEXTURE_BOUND,
            "gt_runtime_information": False,
            "viewer_started": False,
        },
        "combined": aggregate(rows),
        "by_set": {
            set_name: {
                "count": sum(row["set"] == set_name for row in rows),
                **aggregate([row for row in rows if row["set"] == set_name]),
            }
            for set_name, _, _ in SETS
        },
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "texture_guard_changed_count": len(changed),
        "texture_guard_changed_cases": changed,
        "da3_translation": da3_translation,
    }
    output = OUTPUT / "v32_final_autonomous_explicit_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    DOC.write_text(markdown_table(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "combined": report["combined"],
                "texture_guard_changed_count": report[
                    "texture_guard_changed_count"
                ],
                "da3_translation": report["da3_translation"],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")
    print(f">> wrote {DOC}")


if __name__ == "__main__":
    main()
