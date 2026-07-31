#!/usr/bin/env python3
"""EgoHumans CPU-cache evaluation of shared accepted-set SO(3) Kabsch."""

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

from versions.v14 import eval_brtc_global_orientation_kabsch_egohumans as individual_ego  # noqa: E402
from versions.v14 import eval_brtc_multithumbs_egohumans as ego  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14.b0_person_triangulation_shared_orientation_kabsch import (  # noqa: E402
    TORSO4,
    SharedOrientationKabschConfig,
    bounded_rotation,
    kabsch_rotation,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_global_orientation_kabsch/"
    "FROZEN_POLICY_BEFORE_VALIDATION.json"
)
DEFAULT_MULTIHUMAN = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_shared_orientation_kabsch/"
    "HELDOUT_RESULTS.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_shared_orientation_kabsch/egohumans"
)
REPORT_METRICS = individual_ego.REPORT_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--multihuman_report", type=Path, default=DEFAULT_MULTIHUMAN)
    parser.add_argument("--data_root", type=Path, default=ego.DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def shared_boundary_rotation(
    orientation_pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
    config: SharedOrientationKabschConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    pre_centered, post_centered = [], []
    accepted_post_indices = []
    for post_index, record in sorted(records.items()):
        if not bool(record["accepted"]):
            continue
        pre_index = int(record["pre_index"])
        pre_person = orientation_pre_people[pre_index]
        post_person = post_people[post_index]
        pre_joints = np.asarray(pre_person["joints"], dtype=np.float64)
        post_joints = np.asarray(post_person["joints"], dtype=np.float64)
        ids = [index for index in TORSO4 if index < min(len(pre_joints), len(post_joints))]
        if len(ids) < 3:
            return np.eye(3), {
                "applied": False,
                "reason": "insufficient_torso_joints",
                "accepted_post_indices": [],
            }
        pre_root = np.asarray(pre_person["root"], dtype=np.float64)
        post_root = np.asarray(post_person["root"], dtype=np.float64)
        pre_centered.append(pre_joints[ids] - pre_root)
        post_centered.append(post_joints[ids] - post_root)
        accepted_post_indices.append(post_index)
    if not accepted_post_indices:
        return np.eye(3), {
            "applied": False,
            "reason": "no_accepted_people",
            "accepted_post_indices": [],
        }
    pre_stack = np.concatenate(pre_centered, axis=0)
    post_stack = np.concatenate(post_centered, axis=0)
    before = float(np.linalg.norm(post_stack - pre_stack, axis=1).mean())
    raw = kabsch_rotation(post_stack, pre_stack)
    rotation, raw_angle, applied_angle = bounded_rotation(raw, config)
    after = float(np.linalg.norm(post_stack @ rotation.T - pre_stack, axis=1).mean())
    relative = float((before - after) / max(before, 1e-12))
    apply = bool(
        applied_angle > 1e-8
        and after < before
        and relative >= float(config.min_observable_relative_improvement)
    )
    return rotation, {
        "applied": apply,
        "reason": "applied" if apply else "observable_gate",
        "accepted_post_indices": accepted_post_indices,
        "raw_residual_m": before,
        "candidate_residual_m": after,
        "observable_relative_improvement": relative,
        "raw_angle_deg": raw_angle,
        "applied_angle_deg": applied_angle,
        "rotation_world": rotation,
    }


def replay_shared_orientation(
    b0_chains: list[dict[str, Any]],
    brtc_reference_chains: list[dict[str, Any]],
    frozen_boundary_rows: list[dict[str, Any]],
    config: SharedOrientationKabschConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boundary_by_key = {
        (int(row["chain_index"]), int(row["cut_index"])): row
        for row in frozen_boundary_rows
    }
    output_chains, runtime_rows = [], []
    for b0_chain, reference_chain in zip(b0_chains, brtc_reference_chains):
        chain_index = int(b0_chain["chain_index"])
        candidate_segments = [copy.deepcopy(reference_chain["segments"][0])]
        for segment_index in (1, 2):
            b0_post_frames = b0_chain["segments"][segment_index]
            reference_post_frames = reference_chain["segments"][segment_index]
            post_frames = copy.deepcopy(reference_post_frames)
            orientation_pre_frame = candidate_segments[-1][-1]
            orientation_pre_by_track = {
                int(person["global_track_id"]): person
                for person in orientation_pre_frame["people"]
            }
            frozen = boundary_by_key[(chain_index, segment_index - 1)]
            association = frozen["association"]
            track_post_pairs = sorted(association["track_to_post_index"].items())
            orientation_pre_people = [
                orientation_pre_by_track[int(track)] for track, _ in track_post_pairs
            ]
            post_people = post_frames[0]["people"]
            brtc_debug = frozen["brtc"]
            records = individual_ego.debug_by_post_index(brtc_debug)
            rotation, shared = shared_boundary_rotation(
                orientation_pre_people, post_people, records, config
            )
            accepted_post = set(shared["accepted_post_indices"])
            accepted_native = {
                int(post_people[index]["native_track_id"]) for index in accepted_post
            }
            orientation_people = []
            for post_index, person in enumerate(post_people):
                record = records.get(post_index)
                accepted = bool(record is not None and record["accepted"])
                applied = bool(shared["applied"] and accepted)
                orientation_people.append(
                    {
                        "post_index": post_index,
                        "native_track_id": int(person["native_track_id"]),
                        "global_track_id": int(person["global_track_id"]),
                        "brtc_accepted": accepted,
                        "orientation": {
                            "applied": applied,
                            "reason": shared["reason"] if accepted else (
                                "unmatched_exact_b0_fallback"
                                if record is None
                                else "brtc_rejected_exact_b0_fallback"
                            ),
                            "rotation_world": rotation if applied else np.eye(3),
                            "applied_angle_deg": (
                                float(shared.get("applied_angle_deg", 0.0))
                                if applied
                                else 0.0
                            ),
                            "shared_accepted_person_count": len(accepted_post),
                        },
                    }
                )

            fallback_delta = root_delta = camera_delta = 0.0
            post_person_frames = accepted_frames = applied_frames = 0
            for frame, b0_frame, reference_frame in zip(
                post_frames, b0_post_frames, reference_post_frames
            ):
                camera_delta = max(
                    camera_delta,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(frame["method_camera_c2w"], dtype=np.float64)
                                - np.asarray(b0_frame["method_camera_c2w"], dtype=np.float64)
                            )
                        )
                    ),
                )
                b0_by_native = {
                    int(person["native_track_id"]): person for person in b0_frame["people"]
                }
                reference_by_native = {
                    int(person["native_track_id"]): person
                    for person in reference_frame["people"]
                }
                corrected_people = []
                for person in frame["people"]:
                    post_person_frames += 1
                    native = int(person["native_track_id"])
                    corrected = copy.deepcopy(person)
                    if native in accepted_native:
                        accepted_frames += 1
                        if bool(shared["applied"]):
                            corrected = individual_ego.rotate_person_around_root(
                                corrected, rotation
                            )
                            applied_frames += 1
                    else:
                        fallback_delta = max(
                            fallback_delta,
                            individual_ego.maximum_geometry_delta(
                                corrected, b0_by_native[native]
                            ),
                        )
                    root_delta = max(
                        root_delta,
                        individual_ego.maximum_geometry_delta(
                            corrected, reference_by_native[native], keys=("root",)
                        ),
                    )
                    corrected_people.append(corrected)
                frame["people"] = corrected_people
            if fallback_delta > 0.0 or root_delta > 0.0 or camera_delta > 0.0:
                raise ValueError("Shared Kabsch violated fallback/root/camera invariants")

            reference_pre_frame = reference_chain["segments"][segment_index - 1][-1]
            reference_pre_by_track = {
                int(person["global_track_id"]): person
                for person in reference_pre_frame["people"]
            }
            inherited = {
                int(track): float(
                    np.max(
                        np.abs(
                            np.asarray(person["joints"], dtype=np.float64)
                            - np.asarray(reference_pre_by_track[track]["joints"], dtype=np.float64)
                        )
                    )
                )
                for track, person in orientation_pre_by_track.items()
                if track in reference_pre_by_track
            }
            runtime_rows.append(
                {
                    "chain_index": chain_index,
                    "cut_index": segment_index - 1,
                    "association": association,
                    "brtc": brtc_debug,
                    "orientation_people": orientation_people,
                    "shared_orientation": shared,
                    "brtc_first_geometry_parity_max_abs_delta": 0.0,
                    "rejected_unmatched_exact_b0_max_abs_change": fallback_delta,
                    "root_vs_v1_max_abs_delta": root_delta,
                    "camera_vs_b0_max_abs_delta": camera_delta,
                    "post_person_frame_count": post_person_frames,
                    "propagated_brtc_accepted_person_frame_count": accepted_frames,
                    "propagated_orientation_person_frame_count": applied_frames,
                    "pre_inherited_orientation_joint_max_abs_delta_by_global_track": inherited,
                }
            )
            candidate_segments.append(post_frames)
        output_chains.append(
            {
                "chain_index": chain_index,
                "segments": candidate_segments,
                "frames": [frame for segment in candidate_segments for frame in segment],
            }
        )
    return output_chains, runtime_rows


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen BRTC + shared/group SO(3) Kabsch on EgoHumans",
        "",
        "| Method | W | WA | pelvis MPJPE | pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("b0", "brtc_v1", "brtc_shared_kabsch"):
        value = report["methods"][name]["metrics"]
        lines.append(
            f"| {name} | {value['w_mpjpe_mm']:.3f} | {value['wa_mpjpe_mm']:.3f} | "
            f"{value['pelvis_mpjpe_mm']:.3f} | {value['pelvis_mpvpe_mm']:.3f} | "
            f"{value['fixed_world_root_mm']:.3f} | {value['fixed_world_joint_mm']:.3f} | "
            f"{value['fixed_world_vertex_mm']:.3f} | {value['pairwise_root_distance_mm']:.3f} | "
            f"{value['pairwise_root_vector_mm']:.3f} | "
            f"{value['world_root_accel_delta2_mm_per_frame2']:.3f} | "
            f"{value['world_joint_accel_delta2_mm_per_frame2']:.3f} |"
        )
    lines.extend(["", "## Delta versus BRTC v1", ""])
    for key in REPORT_METRICS:
        lines.append(f"- `{key}`: `{report['delta_vs_brtc_v1'][key]:+.3f}`")
    runtime = report["runtime_audit"]
    lines.extend(
        [
            "",
            f"BRTC accepted: `{runtime['brtc_accepted_count']}/{runtime['matched_count']}`; "
            f"shared Kabsch applied people: `{runtime['orientation_applied_count']}/"
            f"{runtime['brtc_accepted_count']}`.",
            f"Rejected/unmatched exact B0: "
            f"`{runtime['rejected_unmatched_exact_b0_max_abs_change']:.3e}`.",
            f"Native root max delta versus v1: `{runtime['root_vs_v1_max_abs_delta']:.3e}`.",
            f"Camera max delta: `{runtime['camera_vs_b0_max_abs_delta']:.3e}`.",
            f"Second cut uses inherited orientation: "
            f"`{runtime['second_cut_inherited_orientation_observed']}`.",
            "",
            f"MultiHuman held-out pass: `{report['decision']['multihuman_pass']}`.",
            f"All Ego requested mean metrics non-regression: "
            f"`{report['decision']['all_requested_means_not_worse']}`.",
            f"Root/joint Accel non-regression: `{report['decision']['accel_not_worse']}`.",
            f"Runtime invariants pass: `{report['decision']['runtime_invariants_pass']}`.",
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
            raise ValueError("All paths must remain in Movie3R under /data")
    ego.run_self_test()
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    if common.canonical_sha256(frozen["policy"]) != frozen["policy_sha256"]:
        raise ValueError("Inherited Kabsch policy checksum mismatch")
    config = SharedOrientationKabschConfig(**frozen["policy"])
    multihuman = json.loads(args.multihuman_report.read_text(encoding="utf-8"))
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    methods, boundary_debug = ego.method_chains(cache)
    candidate, runtime_rows = replay_shared_orientation(
        methods["b0"], methods["b0_brtc_lc"], boundary_debug, config
    )
    _, exo = ego.load_colmap(args.data_root)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    evaluated, error_maps = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("brtc_v1", methods["b0_brtc_lc"]),
        ("brtc_shared_kabsch", candidate),
    ):
        evaluated[name], _ = individual_ego.evaluate_chains(
            chains, args.data_root, exo, vertex_map, joint_regressor, float(args.fps)
        )
        error_maps[name] = individual_ego.fixed_error_maps(
            chains, args.data_root, exo, vertex_map, joint_regressor
        )
    methods_report = {
        name: {
            "metrics": individual_ego.metric_row(value),
            "coverage": float(value["coverage"]),
            "full_report": value,
        }
        for name, value in evaluated.items()
    }
    reference = methods_report["brtc_v1"]["metrics"]
    candidate_metric = methods_report["brtc_shared_kabsch"]["metrics"]
    delta = {key: candidate_metric[key] - reference[key] for key in REPORT_METRICS}
    runtime = individual_ego.rotation_runtime_audit(runtime_rows)
    runtime["shared_boundaries"] = [
        {
            "chain_index": int(row["chain_index"]),
            "cut_index": int(row["cut_index"]),
            "matched_count": int(row["brtc"]["matched_count"]),
            "accepted_count": int(row["brtc"]["accepted_count"]),
            "shared_applied": bool(row["shared_orientation"]["applied"]),
            "shared_reason": row["shared_orientation"]["reason"],
            "shared_accepted_person_count": len(
                row["shared_orientation"]["accepted_post_indices"]
            ),
            "raw_residual_m": row["shared_orientation"].get("raw_residual_m"),
            "candidate_residual_m": row["shared_orientation"].get(
                "candidate_residual_m"
            ),
            "observable_relative_improvement": row["shared_orientation"].get(
                "observable_relative_improvement"
            ),
            "raw_angle_deg": row["shared_orientation"].get("raw_angle_deg"),
            "applied_angle_deg": row["shared_orientation"].get(
                "applied_angle_deg"
            ),
        }
        for row in runtime_rows
    ]
    geometry = ego.geometry_parity_audit(methods["b0_brtc_lc"], candidate)
    camera = ego.camera_exactness_audit(methods["b0"], candidate)
    harm = individual_ego.point_harm_audit(
        error_maps["brtc_v1"], error_maps["brtc_shared_kabsch"]
    )
    multihuman_pass = bool(multihuman["decision"]["all_required_checks_pass"])
    all_means = all(delta[key] <= 1e-12 for key in REPORT_METRICS)
    accel_safe = all(
        delta[key] <= 1e-12
        for key in (
            "world_root_accel_delta2_mm_per_frame2",
            "world_joint_accel_delta2_mm_per_frame2",
        )
    )
    invariant = bool(
        runtime["rejected_unmatched_exact_b0_max_abs_change"] == 0.0
        and runtime["root_vs_v1_max_abs_delta"] == 0.0
        and runtime["camera_vs_b0_max_abs_delta"] == 0.0
        and geometry["max_abs_delta"]["root"] == 0.0
        and geometry["max_abs_delta"]["camera"] == 0.0
        and camera["bit_exact"]
        and runtime["rotation_max_orthogonality_error"] <= 1e-12
        and runtime["rotation_max_determinant_error"] <= 1e-12
        and runtime["second_cut_inherited_orientation_observed"]
    )
    status = (
        "GO_SHARED_ORIENTATION_KABSCH_EGOHUMANS"
        if multihuman_pass and all_means and accel_safe and invariant
        else "NO_GO_SHARED_ORIENTATION_KABSCH_EGOHUMANS"
    )
    report = {
        "experiment": "v14_brtc_shared_orientation_kabsch_egohumans",
        "protocol": {
            "geometry_cache": str(args.geometry_cache),
            "device": "cpu",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "root_update": "frozen BRTC only",
            "translation_orientation_state_separated": True,
            "official_multithumbs_protocol": False,
        },
        "policy_source": str(args.policy),
        "policy": frozen["policy"],
        "policy_sha256": frozen["policy_sha256"],
        "multihuman_report_source": str(args.multihuman_report),
        "methods": methods_report,
        "delta_vs_brtc_v1": delta,
        "runtime_audit": runtime,
        "geometry_vs_v1": geometry,
        "camera_vs_b0": camera,
        "harm_vs_brtc_v1": harm,
        "decision": {
            "multihuman_pass": multihuman_pass,
            "all_requested_means_not_worse": all_means,
            "accel_not_worse": accel_safe,
            "runtime_invariants_pass": invariant,
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
