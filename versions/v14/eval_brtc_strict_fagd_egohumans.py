#!/usr/bin/env python3
"""Replay strict deployable FAGD-0.9 on the existing EgoHumans CPU cache."""

from __future__ import annotations

import argparse
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
from versions.v14 import eval_brtc_group_damping_egohumans as old_fagd  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14.b0_person_triangulation_strict_fagd import (  # noqa: E402
    StrictFAGDConfig,
    refine_matched_people_strict_fagd,
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
DEFAULT_MULTIHUMAN = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_strict_deployable_fagd/"
    "multihuman_report.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_strict_deployable_fagd/egohumans"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--multihuman_report", type=Path, default=DEFAULT_MULTIHUMAN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def strict_callback(alpha: float):
    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        return refine_matched_people_strict_fagd(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            fagd_config=StrictFAGDConfig(alpha=alpha),
        )

    return callback


def full_metrics(value: dict[str, Any]) -> dict[str, float]:
    return {key: float(item) for key, item in value["metrics"].items()}


def markdown(report: dict[str, Any]) -> str:
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
    labels = ("W", "WA", "Root", "Joint", "Vertex", "Pair dist", "Pair vec", "Root Accel", "Joint Accel")
    lines = [
        "# Strict deployable FAGD-0.9 on EgoHumans CPU geometry cache",
        "",
        f"Runtime gate: `{report['strict_runtime_policy']['gate']}`.",
        "The frozen FAGD policy file supplies `alpha=0.9` only; its older, looser gate is not reused.",
        "",
        "| Method | " + " | ".join(labels) + " | Coverage | Harm >5cm |",
        "|---|" + "---:|" * len(labels) + "---:|---:|",
    ]
    for name in ("b0", "brtc_v1", "strict_fagd"):
        value = report["methods"][name]
        metrics = value["metrics"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{metrics[key]:.3f}" for key in keys)
            + f" | {value['coverage']:.1%} | {value['harm_over_5cm_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Strict runtime bit-exact to the earlier FAGD callback on this cache: "
            f"`{report['old_fagd_parity']['bit_parity']}`.",
            "This parity is expected to be false when the older callback acts on an all-accepted "
            "matched subset whose size is smaller than the observable population.",
            f"Strict gate boundaries: `{report['runtime_audit']['strict_gate_boundary_count']}/"
            f"{report['runtime_audit']['boundary_count']}`.",
            f"Camera max delta: `{report['methods']['strict_fagd']['camera_max_abs_change']:.3e}`.",
            "",
            "## Accel caveat",
            "",
            "`Accel` here is a provisional diagnostic on three self-built 15-frame chains. "
            "Each cross-camera cut repeats the same dataset timestamp, and the paper's official "
            "coordinates/fps/aggregation are unpublished. FAGD is a piecewise rigid post-shot "
            "translation, so it can improve spatial errors while worsening the discrete world-root "
            "second difference. It is not a temporal stabilizer.",
            "",
            f"Ego spatial/layout/harm pass: `{report['decision']['ego_spatial_layout_harm_pass']}`.",
            f"Ego world-root Accel non-regression: `{report['decision']['world_root_accel_not_worse']}`.",
            f"General variable-visibility safety pass: `{report['decision']['general_variable_visibility_safe']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (
        args.geometry_cache,
        args.policy,
        args.multihuman_report,
        args.output_dir,
    ):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain under Movie3R on /data")
    ego.run_self_test()
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen FAGD policy checksum mismatch")
    alpha = float(policy["alpha"])
    if alpha != 0.9:
        raise ValueError(f"Expected alpha 0.9, got {alpha}")
    multihuman = json.loads(args.multihuman_report.read_text(encoding="utf-8"))

    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    methods, boundary_debug = ego.method_chains(cache)
    strict_chains, runtime_rows = ego.replay_refinement_variant(
        methods["b0"],
        boundary_debug,
        "b0_brtc_strict_fagd",
        strict_callback(alpha),
        __name__,
    )
    old_chains, _ = ego.replay_refinement_variant(
        methods["b0"],
        boundary_debug,
        "b0_brtc_old_fagd_callback_parity_only",
        old_fagd.group_only_callback(alpha),
        old_fagd.__name__,
    )
    old_parity = ego.geometry_parity_audit(old_chains, strict_chains)

    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    evaluated, roots = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("brtc_v1", methods["b0_brtc_lc"]),
        ("strict_fagd", strict_chains),
    ):
        evaluated[name], roots[name] = old_fagd.evaluate_chains(
            chains, exo, vertex_map, joint_regressor
        )
    harms = {
        name: ego.harm_audit(roots["b0"], roots[name]) if name != "b0" else None
        for name in evaluated
    }
    cameras = {
        "b0": {"max_abs_change": 0.0, "bit_exact": True},
        "brtc_v1": ego.camera_exactness_audit(methods["b0"], methods["b0_brtc_lc"]),
        "strict_fagd": ego.camera_exactness_audit(methods["b0"], strict_chains),
    }
    method_report = {}
    for name, value in evaluated.items():
        harm = harms[name]
        method_report[name] = {
            "metrics": full_metrics(value),
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
            "camera_max_abs_change": float(cameras[name]["max_abs_change"]),
            "full_report": value,
            "harm_audit": harm,
            "camera_audit": cameras[name],
        }

    runtime_audit = ego.refinement_runtime_audit(runtime_rows)
    runtime_audit["strict_gate_boundary_count"] = int(
        sum(
            bool(row["refinement"]["strict_full_one_to_one_all_accepted"])
            for row in runtime_rows
        )
    )
    runtime_audit["exact_v1_fallback_boundary_count"] = int(
        sum(bool(row["refinement"]["exact_v1_fallback"]) for row in runtime_rows)
    )
    runtime_audit["boundaries"] = [
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
            "exact_v1_fallback": bool(row["refinement"]["exact_v1_fallback"]),
        }
        for row in runtime_rows
    ]
    v1 = method_report["brtc_v1"]
    candidate = method_report["strict_fagd"]
    spatial_keys = (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "fixed_world_joint_mm",
        "fixed_world_vertex_mm",
    )
    spatial = all(candidate["metrics"][key] < v1["metrics"][key] - 1e-12 for key in spatial_keys)
    layout = all(
        candidate["metrics"][key] <= v1["metrics"][key] + 1e-12
        for key in ("pairwise_root_distance_mm", "pairwise_root_vector_mm")
    )
    harm_safe = candidate["harm_over_5cm_rate"] <= v1["harm_over_5cm_rate"] + 1e-12
    accel_safe = (
        candidate["metrics"]["world_root_accel_delta2_mm_per_frame2"]
        <= v1["metrics"]["world_root_accel_delta2_mm_per_frame2"] + 1e-12
    )
    variable_safe = bool(multihuman["decision"]["all_variable_bit_exact_v1"])
    status = (
        "GO_STRICT_FAGD_DEPLOYABLE"
        if spatial and layout and harm_safe and accel_safe and variable_safe
        else "NO_GO_STRICT_FAGD_DEPLOYABLE"
    )
    report = {
        "experiment": "v14_strict_deployable_fagd_egohumans",
        "protocol": {
            "geometry_cache": str(args.geometry_cache),
            "human_forward_rerun": False,
            "device": "cpu",
            "official_multithumbs_protocol": False,
            "scope": "three self-built 15-frame EgoHumans 001_legoassemble chains",
            "accel_caveat": (
                "provisional repeated-timestamp short-chain diagnostic; paper formula/"
                "coordinates/fps/aggregation unpublished; spatial FAGD is not a temporal stabilizer"
            ),
        },
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "strict_runtime_policy": {
            "alpha": alpha,
            "alpha_source": str(args.policy),
            "gate": (
                "accepted_count == matched_count == "
                "max(len(pre_people), len(post_people)) > 0"
            ),
            "application": (
                "scale frozen group median only; keep selected individual residual exact; "
                "otherwise return exact frozen BRTC-LC v1 geometry"
            ),
        },
        "multihuman_report_source": str(args.multihuman_report),
        "methods": method_report,
        "runtime_audit": runtime_audit,
        "old_fagd_parity": old_parity,
        "decision": {
            "spatial_improve": spatial,
            "layout_not_worse": layout,
            "harm_not_worse": harm_safe,
            "ego_spatial_layout_harm_pass": bool(spatial and layout and harm_safe),
            "world_root_accel_not_worse": accel_safe,
            "general_variable_visibility_safe": variable_safe,
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
