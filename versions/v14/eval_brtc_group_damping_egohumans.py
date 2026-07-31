#!/usr/bin/env python3
"""CPU-cache EgoHumans validation of frozen full-accept group-only damping.

This script reuses the current-checkpoint geometry cache built by
``eval_brtc_multithumbs_egohumans.py``.  It performs no Human3R forward and
does not load any additional pretrained model.  The frozen policy is selected
only by ``probe_brtc_full_accept_group_damping.py`` on ``three offset0``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import eval_brtc_multithumbs_egohumans as ego  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14.b0_person_triangulation import (  # noqa: E402
    DEFAULT_CONFIG,
    refine_matched_people,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_full_accept_group_damping/"
    "FROZEN_POLICY_BEFORE_HELDOUT.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_full_accept_group_damping/egohumans"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def group_only_callback(alpha: float):
    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        corrected, base_debug = refine_matched_people(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            DEFAULT_CONFIG,
        )
        full_accept = bool(matches) and int(base_debug["accepted_count"]) == int(
            base_debug["matched_count"]
        )
        debug = dict(base_debug)
        records = []
        group = np.asarray(base_debug["group_shift_world"], dtype=np.float64)
        residual_lambda = float(base_debug["selected_residual_lambda"])
        for base_record in base_debug["people"]:
            record = dict(base_record)
            base_final = np.asarray(base_record["final_shift_world"], dtype=np.float64)
            if full_accept and bool(base_record["accepted"]):
                individual = np.asarray(
                    base_record["individual_shift_world"], dtype=np.float64
                )
                new_shift = float(alpha) * group + residual_lambda * (
                    individual - group
                )
                post_index = int(base_record["post_index"])
                for key in ("root", "joints", "vertices"):
                    if key in corrected[post_index]:
                        corrected[post_index][key] = (
                            np.asarray(post_people[post_index][key], dtype=np.float64)
                            + new_shift
                        )
            else:
                new_shift = base_final
            record["base_final_shift_world"] = base_final
            record["final_shift_world"] = new_shift
            record["group_only_alpha"] = float(alpha) if full_accept else 1.0
            records.append(record)
        debug.update(
            {
                "camera_update": "none",
                "full_accept_observable": full_accept,
                "group_damping_applied": full_accept and float(alpha) != 1.0,
                "group_only_alpha": float(alpha) if full_accept else 1.0,
                "base_group_shift_world": group,
                "group_shift_world": float(alpha) * group if full_accept else group,
                "people": records,
            }
        )
        return corrected, debug

    return callback


def evaluate_chains(chains, exo, vertex_map, joint_regressor):
    per_chain, arrays, roots = [], [], []
    for chain in chains:
        result, raw_arrays, root_errors = ego.evaluate_chain(
            chain,
            ego.DEFAULT_DATA,
            exo,
            vertex_map,
            joint_regressor,
            30.0,
        )
        per_chain.append(result)
        arrays.append(raw_arrays)
        roots.append(root_errors)
    return ego.aggregate_method(per_chain, arrays), roots


def metrics_row(report: dict[str, Any]) -> dict[str, float]:
    metric = report["metrics"]
    return {
        key: float(metric[key])
        for key in (
            "w_mpjpe_mm",
            "wa_mpjpe_mm",
            "fixed_world_root_mm",
            "fixed_world_joint_mm",
            "fixed_world_vertex_mm",
            "pairwise_root_distance_mm",
            "pairwise_root_vector_mm",
            "world_root_accel_delta2_mm_per_frame2",
            "ate_m_sim3",
            "identity_switches_mean_per_stream",
        )
    } | {"coverage": float(report["coverage"])}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen full-accept group-only damping on EgoHumans CPU cache",
        "",
        "This is the existing three-chain provisional protocol, not the unpublished "
        "Multi-THuMBS official split.",
        "",
        f"Frozen `alpha={report['policy']['alpha']}`.",
        "",
        "| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | World root Accel | Coverage | Harm >5cm | Camera max Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("b0", "brtc_v1", "group_only"):
        value = report["methods"][name]["metrics"]
        audit = report["methods"][name]
        lines.append(
            f"| {name} | {value['w_mpjpe_mm']:.3f} | {value['wa_mpjpe_mm']:.3f} | "
            f"{value['fixed_world_root_mm']:.3f} | "
            f"{value['fixed_world_joint_mm']:.3f} | "
            f"{value['fixed_world_vertex_mm']:.3f} | "
            f"{value['pairwise_root_distance_mm']:.3f} | "
            f"{value['pairwise_root_vector_mm']:.3f} | "
            f"{value['world_root_accel_delta2_mm_per_frame2']:.3f} | "
            f"{audit['coverage']:.1%} | {audit['harm_over_5cm_rate']:.1%} | "
            f"{audit['camera_max_abs_change']:.1e} |"
        )
    lines.extend(
        [
            "",
            f"Spatial/W/WA improvement over v1: `{report['decision']['spatial_w_wa_improve']}`.",
            f"Layout not worse: `{report['decision']['layout_not_worse']}`.",
            f"Harm not worse: `{report['decision']['harm_not_worse']}`.",
            f"World-root Accel not worse: `{report['decision']['world_root_accel_not_worse']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.geometry_cache, args.policy, args.output_dir):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All inputs/outputs must stay in Movie3R under /data")
    ego.run_self_test()
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen policy checksum mismatch")
    alpha = float(policy["alpha"])
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    methods, boundary_debug = ego.method_chains(cache)
    candidate, runtime_rows = ego.replay_refinement_variant(
        methods["b0"],
        boundary_debug,
        "b0_brtc_full_accept_group_only",
        group_only_callback(alpha),
        __name__,
    )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    evaluated, roots = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("brtc_v1", methods["b0_brtc_lc"]),
        ("group_only", candidate),
    ):
        evaluated[name], roots[name] = evaluate_chains(
            chains, exo, vertex_map, joint_regressor
        )
    harms = {
        name: ego.harm_audit(roots["b0"], roots[name])
        if name != "b0"
        else None
        for name in evaluated
    }
    camera = {
        "b0": {"max_abs_change": 0.0, "bit_exact": True},
        "brtc_v1": ego.camera_exactness_audit(
            methods["b0"], methods["b0_brtc_lc"]
        ),
        "group_only": ego.camera_exactness_audit(methods["b0"], candidate),
    }
    method_report = {}
    for name, value in evaluated.items():
        harm = harms[name]
        method_report[name] = {
            "metrics": metrics_row(value),
            "coverage": float(value["coverage"]),
            "harm_over_5cm_rate": (
                0.0
                if harm is None
                else float(
                    harm["all_person_frames_in_corrected_post_shots"][
                        "harm_over_5cm_rate"
                    ]
                )
            ),
            "camera_max_abs_change": float(camera[name]["max_abs_change"]),
            "full_report": value,
            "harm_audit": harm,
            "camera_audit": camera[name],
        }
    v1 = method_report["brtc_v1"]["metrics"]
    candidate_metrics = method_report["group_only"]["metrics"]
    spatial_keys = (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "fixed_world_joint_mm",
        "fixed_world_vertex_mm",
    )
    spatial = all(candidate_metrics[key] < v1[key] - 1e-12 for key in spatial_keys)
    layout = all(
        candidate_metrics[key] <= v1[key] + 1e-12
        for key in ("pairwise_root_distance_mm", "pairwise_root_vector_mm")
    )
    harm_safe = (
        method_report["group_only"]["harm_over_5cm_rate"]
        <= method_report["brtc_v1"]["harm_over_5cm_rate"] + 1e-12
    )
    accel_safe = (
        candidate_metrics["world_root_accel_delta2_mm_per_frame2"]
        <= v1["world_root_accel_delta2_mm_per_frame2"] + 1e-12
    )
    status = (
        "GO_EGOHUMANS_FULL_ACCEPT_GROUP_ONLY"
        if spatial and layout and harm_safe and accel_safe
        else "NO_GO_EGOHUMANS_FULL_ACCEPT_GROUP_ONLY"
    )
    report = {
        "experiment": "v14_brtc_full_accept_group_only_damping_egohumans",
        "protocol": {
            "geometry_cache": str(args.geometry_cache),
            "human_forward_rerun": False,
            "device": "cpu",
            "official_multithumbs_protocol": False,
            "scope": "three self-built 15-frame EgoHumans 001_legoassemble chains",
        },
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "methods": method_report,
        "runtime_audit": ego.refinement_runtime_audit(runtime_rows),
        "decision": {
            "spatial_w_wa_improve": spatial,
            "layout_not_worse": layout,
            "harm_not_worse": harm_safe,
            "world_root_accel_not_worse": accel_safe,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
