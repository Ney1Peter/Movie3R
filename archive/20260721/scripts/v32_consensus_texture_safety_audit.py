#!/usr/bin/env python3
"""Audit a texture safety bound on the V24 positive-consensus branch."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import (
    angle_deg,
    capped_rotation,
    relative_rotvec,
    safe_gravity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output" / "v32_consensus_texture_safety"
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
)


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid cache {root}: {len(rows)}/{len(output)}")
    return output


def selected_rotation(
    fixed: np.ndarray,
    torso: np.ndarray,
    wide: dict,
    consensus_texture_bound: float | None,
    consensus_cap_deg: float = 60.0,
) -> tuple[np.ndarray, str, dict]:
    vggt = np.asarray(
        wide["windows"]["full_rgb_1p1"]["candidates"]["coarse"]["transform"],
        dtype=np.float32,
    )[:3, :3]
    torso_vector = relative_rotvec(torso, fixed)
    vggt_vector = relative_rotvec(vggt, fixed)
    torso_residual = float(np.degrees(np.linalg.norm(torso_vector)))
    vggt_residual = float(np.degrees(np.linalg.norm(vggt_vector)))
    direction_cosine = float(
        np.dot(torso_vector, vggt_vector)
        / max(np.linalg.norm(torso_vector) * np.linalg.norm(vggt_vector), 1e-9)
    )
    spread = float(wide["windows"]["full_rgb_1p1"]["rotation_consensus"]["spread_deg"])
    texture = float(wide["texture_score"])
    extends = vggt_residual >= torso_residual + 5.0
    large = bool(
        torso_residual >= 30.0
        and extends
        and vggt_residual <= 100.0
        and spread <= 15.0
    )
    consensus = bool(
        torso_residual >= 10.0
        and direction_cosine >= 0.0
        and extends
        and spread <= 5.0
        and vggt_residual <= 100.0
        and (
            consensus_texture_bound is None
            or texture <= consensus_texture_bound
        )
    )
    base = (
        capped_rotation(torso, vggt, 25.0)
        if large
        else (
            capped_rotation(torso, vggt, float(consensus_cap_deg))
            if consensus
            else torso
        )
    )
    low_texture_conflict = bool(
        torso_residual >= 10.0
        and vggt_residual >= torso_residual + 10.0
        and spread <= 5.0
        and direction_cosine < 0.0
        and texture < 0.05
        and vggt_residual <= 100.0
    )
    selected = capped_rotation(base, vggt, 45.0) if low_texture_conflict else base
    branch = (
        "large"
        if large
        else (
            "consensus"
            if consensus
            else ("low_texture_conflict" if low_texture_conflict else "torso")
        )
    )
    return selected, branch, {
        "texture": texture,
        "torso_residual": torso_residual,
        "vggt_residual": vggt_residual,
        "direction_cosine": direction_cosine,
        "spread": spread,
    }


def metrics(rows: list[dict], baseline_key: str) -> dict:
    fixed = np.asarray([row["fixed_error"] for row in rows])
    torso = np.asarray([row["torso_error"] for row in rows])
    baseline = np.asarray([row[baseline_key] for row in rows])
    final = np.asarray([row["final_error"] for row in rows])
    return {
        "count": len(rows),
        "mean": float(final.mean()),
        "p90": float(np.quantile(final, 0.90)),
        "p95": float(np.quantile(final, 0.95)),
        "catastrophic_rate": float(np.mean(final > 45.0)),
        "rescued_vs_fixed": int(np.sum((fixed > 45.0) & (final <= 45.0))),
        "introduced_vs_fixed": int(np.sum((fixed <= 45.0) & (final > 45.0))),
        "rescued_vs_torso": int(np.sum((torso > 45.0) & (final <= 45.0))),
        "introduced_vs_torso": int(np.sum((torso <= 45.0) & (final > 45.0))),
        "harmful_over_5deg_vs_baseline": int(np.sum(final > baseline + 5.0)),
        "improved_over_5deg_vs_baseline": int(np.sum(final + 5.0 < baseline)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = []
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
            loaded.append(
                {
                    "set": set_name,
                    "case_name": name,
                    "source": human["record"]["source"],
                    "fixed": fixed,
                    "torso": torso,
                    "gt": gt,
                    "wide": wide,
                    "fixed_error": angle_deg(fixed, gt),
                    "torso_error": angle_deg(torso, gt),
                }
            )

    results = []
    for bound in (None, 0.03, 0.05, 0.08, 0.10):
        rows = []
        branches: dict[str, int] = {}
        changed = []
        for case in loaded:
            baseline, baseline_branch, _ = selected_rotation(
                case["fixed"], case["torso"], case["wide"], None
            )
            final, branch, diagnostics = selected_rotation(
                case["fixed"], case["torso"], case["wide"], bound
            )
            branches[branch] = branches.get(branch, 0) + 1
            row = {
                **case,
                "v24_error": angle_deg(baseline, case["gt"]),
                "final_error": angle_deg(final, case["gt"]),
                "branch": branch,
                "diagnostics": diagnostics,
            }
            rows.append(row)
            if angle_deg(final, baseline) > 1e-4:
                changed.append(
                    {
                        "set": case["set"],
                        "case_name": case["case_name"],
                        "source": case["source"],
                        "v24_error": row["v24_error"],
                        "final_error": row["final_error"],
                        "diagnostics": diagnostics,
                    }
                )
        results.append(
            {
                "consensus_texture_bound": bound,
                "branch_counts": branches,
                "changed_count": len(changed),
                "changed_cases": changed,
                "overall_vs_v24": metrics(rows, "v24_error"),
                "by_set_vs_v24": {
                    set_name: metrics(
                        [row for row in rows if row["set"] == set_name],
                        "v24_error",
                    )
                    for set_name in sorted({row["set"] for row in rows})
                },
            }
        )

    report = {
        "experiment": "V32 V24 consensus texture safety audit",
        "protocol": {
            "case_count": len(loaded),
            "gt_used_for_offline_development_only": True,
            "holdout5_reserved_for_frozen_validation": True,
        },
        "results": results,
    }
    output = OUTPUT / "v32_consensus_texture_safety_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    "bound": result["consensus_texture_bound"],
                    "changed_count": result["changed_count"],
                    "overall": result["overall_vs_v24"],
                }
                for result in results
            ],
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
