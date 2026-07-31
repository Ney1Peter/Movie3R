#!/usr/bin/env python3
"""Replay frozen angular-safe FAGD on the existing EgoHumans CPU cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import eval_brtc_group_damping_egohumans as group_eval  # noqa: E402
from versions.v14 import eval_brtc_multithumbs_egohumans as ego  # noqa: E402
from versions.v14 import probe_brtc_angular_safe_group_damping as probe  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14.b0_person_triangulation_angular_safe_fagd import (  # noqa: E402
    refine_matched_people_angular_safe_fagd,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_angular_safe_fagd/"
    "FROZEN_POLICY_BEFORE_HELDOUT.json"
)
DEFAULT_HELDOUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_angular_safe_fagd/"
    "HELDOUT_RESULTS.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_angular_safe_fagd/egohumans"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--heldout_report", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def angular_callback(config):
    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        return refine_matched_people_angular_safe_fagd(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            angular_config=config,
        )

    return callback


def metric_subset(value: dict[str, Any]) -> dict[str, float]:
    return {key: float(item) for key, item in value["metrics"].items()}


def markdown(report: dict[str, Any]) -> str:
    policy = report["policy"]
    keys = (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "fixed_world_joint_mm",
        "fixed_world_vertex_mm",
        "pairwise_root_distance_mm",
        "pairwise_root_vector_mm",
        "world_root_accel_delta2_mm_per_frame2",
        "world_joint_accel_delta2_mm_per_frame2",
    )
    labels = (
        "W",
        "WA",
        "Root",
        "Joint",
        "Vertex",
        "Pair dist",
        "Pair vec",
        "Root Accel",
        "Joint Accel",
    )
    lines = [
        "# Frozen angular-safe FAGD on EgoHumans CPU cache",
        "",
        f"Policy: `{policy['statistic']}`, budget `{policy['angular_budget_deg']} deg`.",
        "",
        "| Method | " + " | ".join(labels) + " | Coverage | Harm >5cm |",
        "|---|" + "---:|" * len(labels) + "---:|---:|",
    ]
    for name in ("b0", "brtc_v1", "angular_safe"):
        value = report["methods"][name]
        metric = value["metrics"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{metric[key]:.3f}" for key in keys)
            + f" | {value['coverage']:.1%} | {value['harm_over_5cm_rate']:.1%} |"
        )
    runtime = report["runtime_audit"]
    lines.extend(
        [
            "",
            f"Strict eligible boundaries: `{runtime['strict_gate_boundary_count']}/"
            f"{runtime['boundary_count']}`.",
            f"Damped boundaries: `{runtime['damping_boundary_count']}/"
            f"{runtime['boundary_count']}`.",
            f"Selected alpha counts: `{runtime['alpha_counts']}`.",
            f"Geometry bit-exact to BRTC v1: `{report['v1_geometry_parity']['bit_parity']}`.",
            f"Camera max delta: `{report['methods']['angular_safe']['camera_max_abs_change']:.3e}`.",
            "",
            "## Accel caveat",
            "",
            "This is the same provisional three-chain repeated-timestamp diagnostic, not the "
            "unpublished Multi-THuMBS protocol. Angular-safe FAGD is a boundary spatial "
            "translation, not a temporal stabilizer.",
            "",
            f"Held-out spatial/layout/harm/count-change pass: "
            f"`{report['decision']['heldout_pass']}`.",
            f"Ego spatial/layout/harm pass: `{report['decision']['ego_spatial_layout_harm_pass']}`.",
            f"Ego Accel non-regression: `{report['decision']['ego_accel_not_worse']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (
        args.geometry_cache,
        args.policy,
        args.heldout_report,
        args.output_dir,
    ):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must stay in Movie3R under /data")
    ego.run_self_test()
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen angular-safe policy checksum mismatch")
    heldout = json.loads(args.heldout_report.read_text(encoding="utf-8"))
    config = probe.config_from_policy(policy)
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    methods, boundary_debug = ego.method_chains(cache)
    candidate, runtime_rows = ego.replay_refinement_variant(
        methods["b0"],
        boundary_debug,
        "b0_brtc_angular_safe_fagd",
        angular_callback(config),
        __name__,
    )
    v1_parity = ego.geometry_parity_audit(methods["b0_brtc_lc"], candidate)
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    evaluated, roots = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("brtc_v1", methods["b0_brtc_lc"]),
        ("angular_safe", candidate),
    ):
        evaluated[name], roots[name] = group_eval.evaluate_chains(
            chains, exo, vertex_map, joint_regressor
        )
    method_report = {}
    for name, value in evaluated.items():
        harm = None if name == "b0" else ego.harm_audit(roots["b0"], roots[name])
        camera = (
            {"max_abs_change": 0.0, "bit_exact": True}
            if name == "b0"
            else ego.camera_exactness_audit(
                methods["b0"],
                methods["b0_brtc_lc"] if name == "brtc_v1" else candidate,
            )
        )
        method_report[name] = {
            "metrics": metric_subset(value),
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
            "camera_max_abs_change": float(camera["max_abs_change"]),
            "full_report": value,
            "harm_audit": harm,
            "camera_audit": camera,
        }
    runtime_audit = ego.refinement_runtime_audit(runtime_rows)
    runtime_audit.update(
        {
            "strict_gate_boundary_count": int(
                sum(
                    bool(row["refinement"]["strict_full_one_to_one_all_accepted"])
                    for row in runtime_rows
                )
            ),
            "damping_boundary_count": int(
                sum(
                    bool(row["refinement"]["angular_damping_applied"])
                    for row in runtime_rows
                )
            ),
            "alpha_counts": {
                str(alpha): int(
                    sum(
                        abs(float(row["refinement"]["selected_group_alpha"]) - alpha)
                        <= 1e-12
                        for row in runtime_rows
                    )
                )
                for alpha in config.alpha_values
            },
            "boundaries": [
                {
                    "chain_index": int(row["chain_index"]),
                    "cut_index": int(row["cut_index"]),
                    "previous_observable_count": int(
                        row["refinement"]["previous_observable_count"]
                    ),
                    "current_observable_count": int(
                        row["refinement"]["current_observable_count"]
                    ),
                    "matched_count": int(row["refinement"]["matched_count"]),
                    "accepted_count": int(row["refinement"]["accepted_count"]),
                    "strict_gate": bool(
                        row["refinement"]["strict_full_one_to_one_all_accepted"]
                    ),
                    "selected_alpha": float(
                        row["refinement"]["selected_group_alpha"]
                    ),
                    "damping_applied": bool(
                        row["refinement"]["angular_damping_applied"]
                    ),
                    "budget_satisfied": bool(
                        row["refinement"]["angular_budget_satisfied"]
                    ),
                }
                for row in runtime_rows
            ],
        }
    )
    reference = method_report["brtc_v1"]
    angular = method_report["angular_safe"]
    spatial_keys = (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "fixed_world_joint_mm",
        "fixed_world_vertex_mm",
    )
    spatial = all(
        angular["metrics"][key] < reference["metrics"][key] - 1e-12
        for key in spatial_keys
    )
    layout = all(
        angular["metrics"][key] <= reference["metrics"][key] + 1e-12
        for key in ("pairwise_root_distance_mm", "pairwise_root_vector_mm")
    )
    harm_safe = angular["harm_over_5cm_rate"] <= reference["harm_over_5cm_rate"] + 1e-12
    accel_safe = all(
        angular["metrics"][key] <= reference["metrics"][key] + 1e-12
        for key in (
            "world_root_accel_delta2_mm_per_frame2",
            "world_joint_accel_delta2_mm_per_frame2",
        )
    )
    heldout_pass = heldout["decision"]["status"] == "GO_ANGULAR_SAFE_FAGD_TO_EGO"
    status = (
        "GO_ANGULAR_SAFE_FAGD_DEPLOYABLE"
        if heldout_pass and spatial and layout and harm_safe and accel_safe
        else "NO_GO_ANGULAR_SAFE_FAGD_DEPLOYABLE"
    )
    report = {
        "experiment": "v14_brtc_angular_safe_fagd_egohumans",
        "protocol": {
            "geometry_cache": str(args.geometry_cache),
            "human_forward_rerun": False,
            "device": "cpu",
            "official_multithumbs_protocol": False,
            "scope": "three self-built 15-frame EgoHumans 001_legoassemble chains",
            "candidate_gt_use": "none",
            "future_frames": 0,
            "camera_update": "none",
        },
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "heldout_report_source": str(args.heldout_report),
        "methods": method_report,
        "v1_geometry_parity": v1_parity,
        "runtime_audit": runtime_audit,
        "decision": {
            "heldout_pass": heldout_pass,
            "ego_spatial_improve": spatial,
            "ego_layout_not_worse": layout,
            "ego_harm_not_worse": harm_safe,
            "ego_spatial_layout_harm_pass": bool(spatial and layout and harm_safe),
            "ego_accel_not_worse": accel_safe,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
