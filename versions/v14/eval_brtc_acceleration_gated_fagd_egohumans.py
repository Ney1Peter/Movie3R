#!/usr/bin/env python3
"""EgoHumans CPU-cache test of acceleration-gated strict FAGD-0.9.

The previous-root delta is already observable before each cut and is invariant
to the constant person shift inherited within a shot.  It is attached as
metadata to the last pre frame, allowing the deployable callback to choose
between frozen BRTC and strict FAGD without GT or future post frames.
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
from versions.v14 import eval_brtc_group_damping_egohumans as group_eval  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_strict_fagd import (  # noqa: E402
    refine_matched_people_strict_fagd,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_acceleration_gated_fagd/egohumans"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def attach_previous_root_deltas(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = copy.deepcopy(chains)
    for chain in output:
        for segment in chain["segments"]:
            if len(segment) < 2:
                continue
            previous = {
                int(person["native_track_id"]): person
                for person in segment[-2]["people"]
            }
            for person in segment[-1]["people"]:
                native = int(person["native_track_id"])
                if native in previous:
                    person["pre_previous_root_delta_observable"] = (
                        np.asarray(previous[native]["root"], dtype=np.float64)
                        - np.asarray(person["root"], dtype=np.float64)
                    )
    return output


def acceleration_gated_callback(
    pre_camera,
    post_camera,
    pre_people,
    post_people,
    matches,
):
    matches = tuple((int(first), int(second)) for first, second in matches)
    base, base_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    fagd, fagd_debug = refine_matched_people_strict_fagd(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    history_ok = bool(matches) and all(
        "pre_previous_root_delta_observable" in pre_people[pre_index]
        for pre_index, _ in matches
    )
    strict_ok = bool(fagd_debug["strict_full_one_to_one_all_accepted"])
    if history_ok and strict_ok:
        base_scores, fagd_scores = [], []
        for pre_index, post_index in matches:
            pre_root = np.asarray(pre_people[pre_index]["root"], dtype=np.float64)
            previous_root = pre_root + np.asarray(
                pre_people[pre_index]["pre_previous_root_delta_observable"],
                dtype=np.float64,
            )
            base_scores.append(np.linalg.norm(
                np.asarray(base[post_index]["root"], dtype=np.float64)
                - 2.0 * pre_root
                + previous_root
            ))
            fagd_scores.append(np.linalg.norm(
                np.asarray(fagd[post_index]["root"], dtype=np.float64)
                - 2.0 * pre_root
                + previous_root
            ))
        base_score = float(np.mean(base_scores))
        fagd_score = float(np.mean(fagd_scores))
    else:
        base_score = fagd_score = float("inf")
    apply = bool(history_ok and strict_ok and fagd_score <= base_score)
    corrected = fagd if apply else base
    source_debug = fagd_debug if apply else base_debug
    debug = dict(source_debug)
    debug.update({
        "camera_update": "none",
        "history_ok": history_ok,
        "strict_fagd_eligible": strict_ok,
        "brtc_predicted_acceleration_score_m": base_score,
        "fagd_predicted_acceleration_score_m": fagd_score,
        "acceleration_gate_applied_fagd": apply,
        "matched_count": len(matches),
        "accepted_count": int(base_debug["accepted_count"]),
    })
    return corrected, debug


def metric_subset(value: dict[str, Any]) -> dict[str, float]:
    metric = value["metrics"]
    return {key: float(metric[key]) for key in (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "fixed_world_joint_mm",
        "fixed_world_vertex_mm",
        "pairwise_root_distance_mm",
        "pairwise_root_vector_mm",
        "world_root_accel_delta2_mm_per_frame2",
    )}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Acceleration-gated strict FAGD on EgoHumans",
        "",
        "| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | Root Accel | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("b0", "v1", "candidate"):
        value = report["methods"][name]
        metric = value["metrics"]
        lines.append(
            f"| {name} | {metric['w_mpjpe_mm']:.3f} | {metric['wa_mpjpe_mm']:.3f} | "
            f"{metric['fixed_world_root_mm']:.3f} | {metric['fixed_world_joint_mm']:.3f} | "
            f"{metric['fixed_world_vertex_mm']:.3f} | "
            f"{metric['pairwise_root_distance_mm']:.3f} | "
            f"{metric['pairwise_root_vector_mm']:.3f} | "
            f"{metric['world_root_accel_delta2_mm_per_frame2']:.3f} | "
            f"{value['harm_over_5cm_rate']:.1%} |"
        )
    lines.extend([
        "",
        f"FAGD applied boundaries: `{report['runtime']['fagd_applied_count']}/"
        f"{report['runtime']['boundary_count']}`.",
        f"All eight metrics non-regression: `{report['decision']['all_eight_non_regression']}`.",
        f"Decision: **{report['decision']['status']}**.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.geometry_cache, args.output_dir):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All files must stay in Movie3R under /data")
    ego.run_self_test()
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    methods, boundary_debug = ego.method_chains(cache)
    augmented_b0 = attach_previous_root_deltas(methods["b0"])
    candidate, runtime_rows = ego.replay_refinement_variant(
        augmented_b0,
        boundary_debug,
        "b0_brtc_acceleration_gated_strict_fagd",
        acceleration_gated_callback,
        __name__,
    )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    evaluated, roots = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("v1", methods["b0_brtc_lc"]),
        ("candidate", candidate),
    ):
        evaluated[name], roots[name] = group_eval.evaluate_chains(
            chains, exo, vertex_map, joint_regressor
        )
    method_rows = {}
    for name in ("b0", "v1", "candidate"):
        harm = None if name == "b0" else ego.harm_audit(roots["b0"], roots[name])
        method_rows[name] = {
            "metrics": metric_subset(evaluated[name]),
            "harm_over_5cm_rate": 0.0 if harm is None else float(
                harm["all_person_frames_in_corrected_post_shots"]["harm_over_5cm_rate"]
            ),
            "harm_audit": harm,
        }
    reference = method_rows["v1"]["metrics"]
    candidate_metrics = method_rows["candidate"]["metrics"]
    keys = tuple(reference)
    all_safe = all(candidate_metrics[key] <= reference[key] + 1e-12 for key in keys)
    harm_safe = (
        method_rows["candidate"]["harm_over_5cm_rate"]
        <= method_rows["v1"]["harm_over_5cm_rate"] + 1e-12
    )
    fagd_count = int(sum(
        row["refinement"]["acceleration_gate_applied_fagd"] for row in runtime_rows
    ))
    report = {
        "experiment": "v14_brtc_acceleration_gated_strict_fagd_egohumans",
        "protocol": {
            "geometry_cache": str(args.geometry_cache),
            "device": "cpu",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "candidate_gt_use": "none",
        },
        "methods": method_rows,
        "runtime": {
            "boundary_count": len(runtime_rows),
            "fagd_applied_count": fagd_count,
            "rows": runtime_rows,
        },
        "decision": {
            "all_eight_non_regression": all_safe,
            "harm_non_regression": harm_safe,
            "status": (
                "GO_ACCELERATION_GATED_STRICT_FAGD"
                if all_safe and harm_safe and fagd_count > 0
                else "NO_GO_ACCELERATION_GATED_STRICT_FAGD"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(ego.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
