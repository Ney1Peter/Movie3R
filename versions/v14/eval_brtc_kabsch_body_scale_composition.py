#!/usr/bin/env python3
"""Posthoc no-retuning composition of frozen Kabsch v1 and body-scale v1.

Both constituent policies were frozen independently before this composition.
The two root-centred actions commute algebraically.  This evaluator preserves
the frozen BRTC translation state, propagates the Kabsch orientation state,
and separately propagates the body-scale state.  No parameter is selected or
changed from any composition result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "versions/v14",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from versions.v14 import eval_brtc_global_orientation_kabsch_egohumans as kego  # noqa: E402
from versions.v14 import eval_brtc_multithumbs_egohumans as ego  # noqa: E402
from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_global_orientation_kabsch as kprobe  # noqa: E402
from versions.v14.b0_person_body_scale_consistency import (  # noqa: E402
    BodyScaleConfig,
    config_from_dict as scale_config_from_dict,
    refine_brtc_output_body_scale,
    scale_person_about_root,
)
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_orientation_kabsch import (  # noqa: E402
    OrientationKabschConfig,
    orientation_candidate,
)


DEFAULT_KABSCH_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_global_orientation_kabsch/"
    "FROZEN_POLICY_BEFORE_VALIDATION.json"
)
DEFAULT_SCALE_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_person_body_scale/"
    "FROZEN_POLICY_BEFORE_HELDOUT.json"
)
DEFAULT_EGO_CACHE = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_kabsch_body_scale_composition"
)
DEFAULT_DOC = (
    REPO_ROOT
    / "versions/v14/docs/V14_BRTC_KABSCH_BODY_SCALE_COMPOSITION_20260801.md"
)
STATIC_METRICS = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pelvis_joint_error_m",
    "pelvis_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)
EGO_METRICS = (
    "w_mpjpe_mm",
    "wa_mpjpe_mm",
    "pelvis_mpjpe_mm",
    "pelvis_mpvpe_mm",
    "fixed_world_root_mm",
    "fixed_world_joint_mm",
    "fixed_world_vertex_mm",
    "pairwise_root_distance_mm",
    "pairwise_root_vector_mm",
    "accel_delta2_mm_per_frame2",
    "world_root_accel_delta2_mm_per_frame2",
    "world_joint_accel_delta2_mm_per_frame2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kabsch_policy", type=Path, default=DEFAULT_KABSCH_POLICY)
    parser.add_argument("--scale_policy", type=Path, default=DEFAULT_SCALE_POLICY)
    parser.add_argument("--geometry_cache", type=Path, default=DEFAULT_EGO_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--skip_ego", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def centered_error(
    predicted: np.ndarray,
    predicted_root: np.ndarray,
    target: np.ndarray,
    target_root: np.ndarray,
) -> float:
    count = min(len(predicted), len(target))
    return float(
        np.linalg.norm(
            (predicted[:count] - predicted_root)
            - (target[:count] - target_root),
            axis=1,
        ).mean()
    )


def point_metrics(person: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    output = harness.point_errors(person, target, full=True)
    output.update(
        {
            "pelvis_joint_error_m": centered_error(
                person["joints"], person["root"], target["joints"], target["root"]
            ),
            "pelvis_vertex_error_m": centered_error(
                person["vertices"], person["root"], target["vertices"], target["root"]
            ),
        }
    )
    return output


def root_layout(
    people: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> dict[str, float]:
    distance, vector = [], []
    for first in range(len(people)):
        for second in range(first + 1, len(people)):
            predicted = np.asarray(people[first]["root"]) - np.asarray(
                people[second]["root"]
            )
            target = np.asarray(targets[first]["root"]) - np.asarray(
                targets[second]["root"]
            )
            distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
            vector.append(float(np.linalg.norm(predicted - target)))
    return {
        "pairwise_distance_error_m": finite_mean(distance),
        "pairwise_vector_error_m": finite_mean(vector),
    }


def evaluate_static_split(
    rows: list[dict[str, Any]],
    orientation_config: OrientationKabschConfig,
    scale_config: BodyScaleConfig,
) -> dict[str, Any]:
    prepared = harness.prepare_all(rows)
    collected = {
        name: {metric: [] for metric in STATIC_METRICS}
        for name in ("individual_kabsch", "composition")
    }
    joint_delta, vertex_delta = [], []
    orientation_applied = scale_applied = brtc_accepted = 0
    root_delta = fallback_delta = 0.0
    cases = []
    for case in prepared:
        pre = [person["pre"] for person in case["people"]]
        post = [person["post"] for person in case["people"]]
        matches = [(index, index) for index in range(len(pre))]
        brtc, brtc_debug = refine_matched_people(
            case["pre_camera"], case["post_camera"], pre, post, matches
        )
        accepted = [
            (int(row["pre_index"]), int(row["post_index"]))
            for row in brtc_debug["people"]
            if bool(row["accepted"])
        ]
        brtc_accepted += len(accepted)
        kabsch = [copy.deepcopy(person) for person in brtc]
        orientation_rows = []
        accepted_post = {post_index for _, post_index in accepted}
        for pre_index, post_index in accepted:
            kabsch[post_index], debug = orientation_candidate(
                pre[pre_index], post[post_index], kabsch[post_index], orientation_config
            )
            orientation_applied += int(bool(debug["applied"]))
            orientation_rows.append({"post_index": post_index, **debug})
        composition, scale_debug = refine_brtc_output_body_scale(
            pre, kabsch, accepted, scale_config
        )
        scale_applied += sum(
            abs(float(row["scale_factor"]) - 1.0) > 1e-12
            for row in scale_debug["people"]
        )
        targets = [person["target"] for person in case["people"]]
        layouts = {
            "individual_kabsch": root_layout(kabsch, targets),
            "composition": root_layout(composition, targets),
        }
        people_rows = []
        for index, (first, second, target) in enumerate(
            zip(kabsch, composition, targets)
        ):
            first_metrics = point_metrics(first, target)
            second_metrics = point_metrics(second, target)
            joint_delta.append(
                second_metrics["joint_error_m"] - first_metrics["joint_error_m"]
            )
            vertex_delta.append(
                second_metrics["vertex_error_m"] - first_metrics["vertex_error_m"]
            )
            for name, values in (
                ("individual_kabsch", first_metrics),
                ("composition", second_metrics),
            ):
                for metric in (
                    "root_error_m",
                    "joint_error_m",
                    "vertex_error_m",
                    "pelvis_joint_error_m",
                    "pelvis_vertex_error_m",
                ):
                    collected[name][metric].append(values[metric])
            root_delta = max(
                root_delta,
                float(
                    np.max(
                        np.abs(
                            np.asarray(first["root"]) - np.asarray(second["root"])
                        )
                    )
                ),
            )
            if index not in accepted_post:
                for key in ("root", "joints", "vertices"):
                    fallback_delta = max(
                        fallback_delta,
                        float(
                            np.max(
                                np.abs(np.asarray(second[key]) - np.asarray(brtc[index][key]))
                            )
                        ),
                    )
            people_rows.append(
                {
                    "identity_evaluation_only": case["people"][index]["identity"],
                    "individual_kabsch": first_metrics,
                    "composition": second_metrics,
                }
            )
        for name in ("individual_kabsch", "composition"):
            for metric in ("pairwise_distance_error_m", "pairwise_vector_error_m"):
                collected[name][metric].append(layouts[name][metric])
        cases.append(
            {
                "case": case["case"],
                "orientation": orientation_rows,
                "scale": scale_debug,
                "people": people_rows,
            }
        )
    metrics = {
        name: {key: finite_mean(value) for key, value in rows.items()}
        for name, rows in collected.items()
    }
    delta = {
        key: metrics["composition"][key] - metrics["individual_kabsch"][key]
        for key in STATIC_METRICS
    }
    return {
        "case_count": len(prepared),
        "person_count": sum(len(case["people"]) for case in prepared),
        "brtc_accepted_count": brtc_accepted,
        "orientation_applied_count": orientation_applied,
        "scale_applied_count": scale_applied,
        "individual_kabsch": metrics["individual_kabsch"],
        "composition": metrics["composition"],
        "delta_composition_minus_kabsch": delta,
        "joint_incremental_harm_over_1cm_rate": float(
            np.mean(np.asarray(joint_delta) > 0.01)
        ),
        "joint_incremental_harm_over_5cm_rate": float(
            np.mean(np.asarray(joint_delta) > 0.05)
        ),
        "vertex_incremental_harm_over_1cm_rate": float(
            np.mean(np.asarray(vertex_delta) > 0.01)
        ),
        "vertex_incremental_harm_over_5cm_rate": float(
            np.mean(np.asarray(vertex_delta) > 0.05)
        ),
        "native_root_max_abs_delta_vs_kabsch": root_delta,
        "rejected_unmatched_max_abs_delta_vs_b0": fallback_delta,
        "cases": cases,
    }


def replay_scale_over_kabsch(
    kabsch_chains: list[dict[str, Any]],
    orientation_rows: list[dict[str, Any]],
    config: BodyScaleConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Second causal state: propagate frozen scale over frozen Kabsch shots."""

    rows_by_key = {
        (int(row["chain_index"]), int(row["cut_index"])): row
        for row in orientation_rows
    }
    output, runtime = [], []
    for chain in kabsch_chains:
        chain_index = int(chain["chain_index"])
        base_segments = chain["segments"]
        candidate_segments = [copy.deepcopy(base_segments[0])]
        for segment_index in (1, 2):
            post_frames = copy.deepcopy(base_segments[segment_index])
            pre_people = list(candidate_segments[-1][-1]["people"])
            pre_by_track = {
                int(person["global_track_id"]): index
                for index, person in enumerate(pre_people)
            }
            frozen = rows_by_key[(chain_index, segment_index - 1)]
            accepted_matches = []
            for track, post_index in sorted(
                frozen["association"]["track_to_post_index"].items()
            ):
                native = int(post_frames[0]["people"][int(post_index)]["native_track_id"])
                action = frozen["action_by_native_track"].get(native)
                if action is not None and bool(action["brtc_accepted"]):
                    accepted_matches.append((pre_by_track[int(track)], int(post_index)))
            corrected_first, debug = refine_brtc_output_body_scale(
                pre_people, post_frames[0]["people"], accepted_matches, config
            )
            scale_by_native = {
                int(post_frames[0]["people"][int(row["post_index"])]["native_track_id"]): float(
                    row["scale_factor"]
                )
                for row in debug["people"]
            }
            for frame in post_frames:
                frame["people"] = [
                    scale_person_about_root(
                        person,
                        scale_by_native.get(int(person["native_track_id"]), 1.0),
                    )
                    for person in frame["people"]
                ]
            first_delta = 0.0
            for expected, observed in zip(corrected_first, post_frames[0]["people"]):
                for key in ("root", "joints", "vertices"):
                    first_delta = max(
                        first_delta,
                        float(
                            np.max(
                                np.abs(np.asarray(expected[key]) - np.asarray(observed[key]))
                            )
                        ),
                    )
            runtime.append(
                {
                    "chain_index": chain_index,
                    "cut_index": segment_index - 1,
                    "scale": debug,
                    "scale_by_native_track": scale_by_native,
                    "first_frame_replay_max_abs_delta": first_delta,
                    "inherited_scaled_pre": bool(segment_index == 2),
                }
            )
            candidate_segments.append(post_frames)
        output.append(
            {
                "chain_index": chain_index,
                "segments": candidate_segments,
                "frames": [frame for segment in candidate_segments for frame in segment],
            }
        )
    return output, runtime


def evaluate_ego_chains(chains, exo, vertex_map, joint_regressor):
    results, arrays, roots = [], [], []
    for chain in chains:
        result, raw_arrays, root_errors = ego.evaluate_chain(
            chain, ego.DEFAULT_DATA, exo, vertex_map, joint_regressor, 30.0
        )
        results.append(result)
        arrays.append(raw_arrays)
        roots.append(root_errors)
    return ego.aggregate_method(results, arrays), arrays, roots


def array_harm(reference, candidate):
    output = {}
    for label, key in (
        ("fixed_joint", "fixed_joint_m"),
        ("fixed_vertex", "fixed_vertex_m"),
        ("pelvis_mpjpe", "pelvis_mpjpe_m"),
        ("pelvis_mpvpe", "pelvis_mpvpe_m"),
    ):
        first = np.concatenate([row[key] for row in reference])
        second = np.concatenate([row[key] for row in candidate])
        delta = second - first
        output[label] = {
            "count": len(delta),
            "mean_delta_mm": float(delta.mean() * 1000.0),
            "improve_rate": float(np.mean(delta < 0.0)),
            "harm_over_1cm_rate": float(np.mean(delta > 0.01)),
            "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        }
    return output


def evaluate_ego(
    cache_path: Path,
    orientation_config: OrientationKabschConfig,
    probe_policy: kprobe.OrientationPolicy,
    scale_config: BodyScaleConfig,
) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    methods, boundary_rows = ego.method_chains(cache)
    kabsch_chains, orientation_runtime = kego.replay_brtc_then_orientation(
        methods["b0"],
        methods["b0_brtc_lc"],
        boundary_rows,
        orientation_config,
        probe_policy,
    )
    composition_chains, scale_runtime = replay_scale_over_kabsch(
        kabsch_chains, orientation_runtime, scale_config
    )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    reports, arrays, roots = {}, {}, {}
    for name, chains in (
        ("brtc_v1", methods["b0_brtc_lc"]),
        ("individual_kabsch", kabsch_chains),
        ("kabsch_body_scale", composition_chains),
    ):
        reports[name], arrays[name], roots[name] = evaluate_ego_chains(
            chains, exo, vertex_map, joint_regressor
        )
    metrics = {
        name: {key: float(report["metrics"][key]) for key in EGO_METRICS}
        for name, report in reports.items()
    }
    incremental = {
        key: metrics["kabsch_body_scale"][key] - metrics["individual_kabsch"][key]
        for key in EGO_METRICS
    }
    net = {
        key: metrics["kabsch_body_scale"][key] - metrics["brtc_v1"][key]
        for key in EGO_METRICS
    }
    root_delta = max(
        float(
            np.max(
                np.abs(
                    np.asarray(first["root"])
                    - np.asarray(second["root"])
                )
            )
        )
        for first_chain, second_chain in zip(kabsch_chains, composition_chains)
        for first_frame, second_frame in zip(first_chain["frames"], second_chain["frames"])
        for first, second in zip(first_frame["people"], second_frame["people"])
    )
    return {
        "metrics": metrics,
        "incremental_delta_composition_minus_kabsch": incremental,
        "net_delta_composition_minus_brtc": net,
        "incremental_person_harm": array_harm(
            arrays["individual_kabsch"], arrays["kabsch_body_scale"]
        ),
        "native_root_max_abs_delta_vs_kabsch": root_delta,
        "camera_exactness": ego.camera_exactness_audit(
            kabsch_chains, composition_chains
        ),
        "orientation_runtime": kego.rotation_runtime_audit(orientation_runtime),
        "scale_runtime": {
            "boundary_count": len(scale_runtime),
            "first_frame_replay_max_abs_delta": max(
                row["first_frame_replay_max_abs_delta"] for row in scale_runtime
            ),
            "scale_count": sum(
                len(row["scale_by_native_track"]) for row in scale_runtime
            ),
            "nonidentity_scale_count": sum(
                abs(value - 1.0) > 1e-12
                for row in scale_runtime
                for value in row["scale_by_native_track"].values()
            ),
            "scales": [
                value
                for row in scale_runtime
                for value in row["scale_by_native_track"].values()
            ],
            "second_cut_inherited_scale_state": all(
                row["inherited_scaled_pre"]
                for row in scale_runtime
                if int(row["cut_index"]) == 1
            ),
        },
        "runtime_rows": {
            "orientation": orientation_runtime,
            "scale": scale_runtime,
        },
    }


def static_pass(value: dict[str, Any]) -> bool:
    delta = value["delta_composition_minus_kabsch"]
    return bool(
        all(delta[key] <= 1e-12 for key in STATIC_METRICS)
        and value["joint_incremental_harm_over_5cm_rate"] <= 0.05
        and value["vertex_incremental_harm_over_5cm_rate"] <= 0.05
        and value["native_root_max_abs_delta_vs_kabsch"] == 0.0
        and value["rejected_unmatched_max_abs_delta_vs_b0"] == 0.0
    )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen individual Kabsch v1 + frozen body-scale v1 composition",
        "",
        "Posthoc composition only: no parameter was selected or changed.",
        "",
        f"Kabsch policy SHA256: `{report['frozen_inputs']['kabsch_file_sha256']}`; "
        f"body-scale policy SHA256: `{report['frozen_inputs']['scale_file_sha256']}`.",
        "",
        "## MultiHuman incremental result versus individual Kabsch",
        "",
        "| Split | Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["multihuman"].items():
        for name in ("individual_kabsch", "composition"):
            row = value[name]
            lines.append(
                f"| {split} | {name} | {row['root_error_m']:.6f} | "
                f"{row['joint_error_m']:.6f} | {row['vertex_error_m']:.6f} | "
                f"{row['pelvis_joint_error_m']:.6f} | {row['pelvis_vertex_error_m']:.6f} | "
                f"{row['pairwise_distance_error_m']:.6f} | {row['pairwise_vector_error_m']:.6f} |"
            )
        lines.append(
            f"| {split} | pass |  |  |  |  |  |  | {report['multihuman_pass'][split]} |"
        )
    if "egohumans" in report:
        lines.extend(
            [
                "",
                "## EgoHumans",
                "",
                "| Method | W | WA | Pelvis MPJPE | Pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("brtc_v1", "individual_kabsch", "kabsch_body_scale"):
            row = report["egohumans"]["metrics"][name]
            lines.append(
                f"| {name} | {row['w_mpjpe_mm']:.3f} | {row['wa_mpjpe_mm']:.3f} | "
                f"{row['pelvis_mpjpe_mm']:.3f} | {row['pelvis_mpvpe_mm']:.3f} | "
                f"{row['fixed_world_root_mm']:.3f} | {row['fixed_world_joint_mm']:.3f} | "
                f"{row['fixed_world_vertex_mm']:.3f} | {row['pairwise_root_distance_mm']:.3f} | "
                f"{row['pairwise_root_vector_mm']:.3f} | "
                f"{row['world_root_accel_delta2_mm_per_frame2']:.3f} | "
                f"{row['world_joint_accel_delta2_mm_per_frame2']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"- Incremental delta composition-Kabsch: `{report['egohumans']['incremental_delta_composition_minus_kabsch']}`.",
                f"- Net delta composition-BRTC: `{report['egohumans']['net_delta_composition_minus_brtc']}`.",
                f"- Incremental person harm: `{report['egohumans']['incremental_person_harm']}`.",
                f"- Ego all-gate pass vs individual Kabsch: `{report['egohumans_pass']}`.",
            ]
        )
    lines.extend(
        [
            "",
            f"- Overall qualified second candidate: **{report['qualified_second_candidate']}**.",
            f"- Decision: **{report['decision']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (
        args.kabsch_policy,
        args.scale_policy,
        args.geometry_cache,
        args.output_dir,
        args.doc.parent,
    ):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain inside Movie3R /data")
    kabsch_file_sha = sha256(args.kabsch_policy)
    scale_file_sha = sha256(args.scale_policy)
    kabsch_frozen = json.loads(args.kabsch_policy.read_text(encoding="utf-8"))
    scale_frozen = json.loads(args.scale_policy.read_text(encoding="utf-8"))
    orientation_config = OrientationKabschConfig(**kabsch_frozen["policy"])
    probe_policy = kprobe.OrientationPolicy(**kabsch_frozen["policy"])
    scale_config = scale_config_from_dict(scale_frozen["policy"])
    split_rows = {
        "three_offset1": harness.load_rows(
            "confirm", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
        ),
        "dance": legacy.report_rows(("dance",))[: int(args.max_cases) or None],
        "box": legacy.report_rows(("box",))[: int(args.max_cases) or None],
    }
    multihuman = {
        name: evaluate_static_split(rows, orientation_config, scale_config)
        for name, rows in split_rows.items()
    }
    multihuman_pass = {name: static_pass(value) for name, value in multihuman.items()}
    report = {
        "experiment": "v14_frozen_kabsch_body_scale_posthoc_composition",
        "protocol": {
            "status": "posthoc composition; no retuning and no new policy",
            "order": "BRTC translation -> individual Kabsch -> body scale",
            "commutativity": "rotation and uniform scale share native root pivot",
            "causal_states": "frozen BRTC translation reference; local orientation+scale history",
            "runtime_inputs": "last-pre/current-post predictions only",
        },
        "frozen_inputs": {
            "kabsch_policy": str(args.kabsch_policy),
            "kabsch_file_sha256": kabsch_file_sha,
            "kabsch_config": kabsch_frozen["policy"],
            "scale_policy": str(args.scale_policy),
            "scale_file_sha256": scale_file_sha,
            "scale_config": scale_frozen["policy"],
        },
        "multihuman": multihuman,
        "multihuman_pass": multihuman_pass,
    }
    if not args.skip_ego:
        report["egohumans"] = evaluate_ego(
            args.geometry_cache,
            orientation_config,
            probe_policy,
            scale_config,
        )
    ego_pass = True
    if "egohumans" in report:
        incremental = report["egohumans"]["incremental_delta_composition_minus_kabsch"]
        harm = report["egohumans"]["incremental_person_harm"]
        ego_pass = bool(
            all(incremental[key] <= 1e-12 for key in EGO_METRICS)
            and harm["fixed_joint"]["harm_over_5cm_rate"] <= 0.05
            and harm["fixed_vertex"]["harm_over_5cm_rate"] <= 0.05
            and report["egohumans"]["native_root_max_abs_delta_vs_kabsch"] == 0.0
            and report["egohumans"]["camera_exactness"]["bit_exact"]
        )
    report["egohumans_pass"] = ego_pass
    report["qualified_second_candidate"] = bool(
        all(multihuman_pass.values()) and ego_pass
    )
    report["decision"] = (
        "QUALIFIED_SECOND_CANDIDATE"
        if report["qualified_second_candidate"]
        else "NO_GO_POSTHOC_COMPOSITION"
    )
    report["frozen_inputs_after"] = {
        "kabsch_file_sha256": sha256(args.kabsch_policy),
        "scale_file_sha256": sha256(args.scale_policy),
        "unchanged": bool(
            sha256(args.kabsch_policy) == kabsch_file_sha
            and sha256(args.scale_policy) == scale_file_sha
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "REPORT.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    args.doc.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
