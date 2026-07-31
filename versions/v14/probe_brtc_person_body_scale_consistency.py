#!/usr/bin/env python3
"""Develop, freeze, and blindly validate BRTC person-local body scale."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
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

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14.b0_person_body_scale_consistency import (  # noqa: E402
    BodyScaleConfig,
    config_dict,
    config_from_dict,
    refine_brtc_output_body_scale,
    refine_matched_people_body_scale_consistency,
    scale_person_about_root,
)
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_person_body_scale"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
DEFAULT_EGO_CACHE = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
METRICS = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pelvis_joint_error_m",
    "pelvis_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "freeze", "validate"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--confirm_report", type=Path, default=harness.DEFAULT_CONFIRM_REPORT
    )
    parser.add_argument("--ego_geometry_cache", type=Path, default=DEFAULT_EGO_CACHE)
    parser.add_argument("--skip_ego", action="store_true")
    parser.add_argument("--max_cases", type=int, default=0)
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


def prepare_case_runtime(case: dict[str, Any]) -> dict[str, Any]:
    pre = [person["pre"] for person in case["people"]]
    post = [person["post"] for person in case["people"]]
    matches = [(index, index) for index in range(len(pre))]
    brtc_people, brtc_debug = refine_matched_people(
        case["pre_camera"], case["post_camera"], pre, post, matches
    )
    accepted = [
        (int(row["pre_index"]), int(row["post_index"]))
        for row in brtc_debug["people"]
        if bool(row["accepted"])
    ]
    return {
        "case": case,
        "pre_people": pre,
        "brtc_people": brtc_people,
        "brtc_debug": brtc_debug,
        "accepted_matches": accepted,
    }


def prepare_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = harness.prepare_all(rows)
    output = []
    for index, case in enumerate(cases, start=1):
        output.append(prepare_case_runtime(case))
        print(
            f"[{index:03d}/{len(cases):03d}] cached BRTC {case['case']['key']}",
            flush=True,
        )
    return output


def person_metrics(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, float]:
    output = harness.point_errors(predicted, target, full=True)
    output.update(
        {
            "pelvis_joint_error_m": centered_error(
                predicted["joints"],
                predicted["root"],
                target["joints"],
                target["root"],
            ),
            "pelvis_vertex_error_m": centered_error(
                predicted["vertices"],
                predicted["root"],
                target["vertices"],
                target["root"],
            ),
        }
    )
    return output


def layout(
    roots: list[np.ndarray], target_roots: list[np.ndarray]
) -> dict[str, float]:
    distance, vector = [], []
    for first in range(len(roots)):
        for second in range(first + 1, len(roots)):
            predicted = roots[first] - roots[second]
            target = target_roots[first] - target_roots[second]
            distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
            vector.append(float(np.linalg.norm(predicted - target)))
    return {
        "pairwise_distance_error_m": finite_mean(distance),
        "pairwise_vector_error_m": finite_mean(vector),
    }


def evaluate_config(
    records: list[dict[str, Any]], config: BodyScaleConfig, include_cases: bool
) -> dict[str, Any]:
    values = {metric: [] for metric in METRICS}
    baseline = {metric: [] for metric in METRICS}
    joint_delta, vertex_delta = [], []
    root_max_abs_change = 0.0
    rejected_or_unmatched_max_abs_change = 0.0
    scale_factors, mad_values, valid_counts = [], [], []
    fallback_reasons: Counter[str] = Counter()
    case_rows = []
    for record in records:
        case = record["case"]
        corrected, debug = refine_brtc_output_body_scale(
            record["pre_people"],
            record["brtc_people"],
            record["accepted_matches"],
            config,
        )
        root_max_abs_change = max(
            root_max_abs_change, float(debug["root_max_abs_change"])
        )
        scaled_post_indices = {
            int(row["post_index"]) for row in debug["people"]
        }
        roots, base_roots, target_roots = [], [], []
        people_rows = []
        debug_by_post = {int(row["post_index"]): row for row in debug["people"]}
        for person_index, (candidate, current, person) in enumerate(
            zip(corrected, record["brtc_people"], case["people"])
        ):
            candidate_metrics = person_metrics(candidate, person["target"])
            current_metrics = person_metrics(current, person["target"])
            for metric in (
                "root_error_m",
                "joint_error_m",
                "vertex_error_m",
                "pelvis_joint_error_m",
                "pelvis_vertex_error_m",
            ):
                values[metric].append(candidate_metrics[metric])
                baseline[metric].append(current_metrics[metric])
            joint_delta.append(
                candidate_metrics["joint_error_m"] - current_metrics["joint_error_m"]
            )
            vertex_delta.append(
                candidate_metrics["vertex_error_m"] - current_metrics["vertex_error_m"]
            )
            roots.append(np.asarray(candidate["root"]))
            base_roots.append(np.asarray(current["root"]))
            target_roots.append(np.asarray(person["target"]["root"]))
            if person_index not in scaled_post_indices:
                for key in ("root", "joints", "vertices"):
                    rejected_or_unmatched_max_abs_change = max(
                        rejected_or_unmatched_max_abs_change,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(candidate[key]) - np.asarray(current[key])
                                )
                            )
                        ),
                    )
            if person_index in debug_by_post:
                row = debug_by_post[person_index]
                scale_factors.append(float(row["scale_factor"]))
                mad_values.append(float(row["evidence"]["log_ratio_mad"]))
                valid_counts.append(int(row["evidence"]["valid_edge_count"]))
                if not bool(row["accepted"]):
                    fallback_reasons[str(row["fallback_reason"])] += 1
            if include_cases:
                people_rows.append(
                    {
                        "identity_evaluation_only": person["identity"],
                        "candidate": candidate_metrics,
                        "brtc_v1": current_metrics,
                        "scale": debug_by_post.get(person_index),
                    }
                )
        candidate_layout = layout(roots, target_roots)
        current_layout = layout(base_roots, target_roots)
        for metric in ("pairwise_distance_error_m", "pairwise_vector_error_m"):
            values[metric].append(candidate_layout[metric])
            baseline[metric].append(current_layout[metric])
        if include_cases:
            case_rows.append(
                {
                    "case": case["case"],
                    "people": people_rows,
                    "candidate_layout": candidate_layout,
                    "brtc_v1_layout": current_layout,
                }
            )
    candidate_mean = {key: finite_mean(value) for key, value in values.items()}
    baseline_mean = {key: finite_mean(value) for key, value in baseline.items()}
    delta = {key: candidate_mean[key] - baseline_mean[key] for key in METRICS}
    factors = np.asarray(scale_factors, dtype=np.float64)
    output = {
        "case_count": len(records),
        "person_count": sum(len(row["case"]["people"]) for row in records),
        "candidate": candidate_mean,
        "brtc_v1": baseline_mean,
        "delta_vs_brtc_v1": delta,
        "joint_person_harm_over_1cm_rate": float(np.mean(np.asarray(joint_delta) > 0.01)),
        "joint_person_harm_over_5cm_rate": float(np.mean(np.asarray(joint_delta) > 0.05)),
        "vertex_person_harm_over_1cm_rate": float(np.mean(np.asarray(vertex_delta) > 0.01)),
        "vertex_person_harm_over_5cm_rate": float(np.mean(np.asarray(vertex_delta) > 0.05)),
        "joint_person_improve_rate": float(np.mean(np.asarray(joint_delta) < 0.0)),
        "vertex_person_improve_rate": float(np.mean(np.asarray(vertex_delta) < 0.0)),
        "scale_evaluated_count": len(scale_factors),
        "nonidentity_scale_count": int(np.sum(np.abs(factors - 1.0) > 1e-12)),
        "scale_factor_mean": finite_mean(factors),
        "scale_factor_min": float(factors.min()) if len(factors) else float("nan"),
        "scale_factor_max": float(factors.max()) if len(factors) else float("nan"),
        "fallback_count": int(sum(fallback_reasons.values())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "log_mad_mean": finite_mean(mad_values),
        "valid_edge_count_mean": finite_mean(valid_counts),
        "root_max_abs_change_vs_brtc": root_max_abs_change,
        "rejected_or_unmatched_max_abs_change_vs_brtc": (
            rejected_or_unmatched_max_abs_change
        ),
    }
    if include_cases:
        output["cases"] = case_rows
    return output


def eligible(summary: dict[str, Any]) -> bool:
    delta = summary["delta_vs_brtc_v1"]
    return bool(
        summary["nonidentity_scale_count"] > 0
        and delta["joint_error_m"] < -1e-12
        and delta["vertex_error_m"] < -1e-12
        and delta["pelvis_joint_error_m"] <= 1e-12
        and delta["pelvis_vertex_error_m"] <= 1e-12
        and abs(delta["root_error_m"]) <= 1e-12
        and abs(delta["pairwise_distance_error_m"]) <= 1e-12
        and abs(delta["pairwise_vector_error_m"]) <= 1e-12
        and summary["joint_person_harm_over_5cm_rate"] <= 0.05
        and summary["vertex_person_harm_over_5cm_rate"] <= 0.05
        and summary["root_max_abs_change_vs_brtc"] == 0.0
        and summary["rejected_or_unmatched_max_abs_change_vs_brtc"] == 0.0
    )


def grid() -> list[BodyScaleConfig]:
    base = BodyScaleConfig(min_valid_edges=12)
    return [
        replace(
            base,
            fraction=fraction,
            relative_cap=cap,
            max_log_mad=mad,
        )
        for fraction in (0.25, 0.50, 0.75, 1.0)
        for cap in (0.05, 0.10, 0.20)
        for mad in (0.01, 0.02, 0.03)
    ]


def summary_row(summary: dict[str, Any]) -> str:
    candidate, current = summary["candidate"], summary["brtc_v1"]
    return (
        f"| BRTC v1 | {current['root_error_m']:.6f} | {current['joint_error_m']:.6f} | "
        f"{current['vertex_error_m']:.6f} | {current['pelvis_joint_error_m']:.6f} | "
        f"{current['pelvis_vertex_error_m']:.6f} | {current['pairwise_distance_error_m']:.6f} | "
        f"{current['pairwise_vector_error_m']:.6f} | 0.0% | 0.0% |\n"
        f"| body-scale | {candidate['root_error_m']:.6f} | {candidate['joint_error_m']:.6f} | "
        f"{candidate['vertex_error_m']:.6f} | {candidate['pelvis_joint_error_m']:.6f} | "
        f"{candidate['pelvis_vertex_error_m']:.6f} | {candidate['pairwise_distance_error_m']:.6f} | "
        f"{candidate['pairwise_vector_error_m']:.6f} | "
        f"{summary['joint_person_harm_over_5cm_rate']:.1%} | "
        f"{summary['vertex_person_harm_over_5cm_rate']:.1%} |"
    )


def table(title: str, summary: dict[str, Any]) -> list[str]:
    return [
        f"## {title}",
        "",
        "| Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec | Joint harm >5cm | Vertex harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row(summary),
        "",
        f"- Scale actions: `{summary['nonidentity_scale_count']}/{summary['scale_evaluated_count']}`; "
        f"range `{summary['scale_factor_min']:.6f}..{summary['scale_factor_max']:.6f}`; "
        f"fallback `{summary['fallback_reasons']}`.",
        f"- Root max change: `{summary['root_max_abs_change_vs_brtc']:.3e}`; "
        f"rejected/unmatched max change: `{summary['rejected_or_unmatched_max_abs_change_vs_brtc']:.3e}`.",
        "",
    ]


def dev_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BRTC person-local robust body-scale: development",
        "",
        "Only `three offset0` was read. No held-out report was loaded.",
        "",
        f"- Eligible: `{report['eligible_count']}/{report['grid_count']}`.",
        f"- Selected: `{report['selected']['config']}`.",
        f"- Decision: **{'GO_TO_FREEZE' if report['dev_pass'] else 'NO_GO'}**.",
        "",
    ]
    lines.extend(table("Selected offset0 result", report["selected"]["summary"]))
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> None:
    rows = harness.load_rows("dev", args.confirm_report, args.max_cases)
    records = prepare_records(rows)
    scan = []
    for index, config in enumerate(grid(), start=1):
        summary = evaluate_config(records, config, include_cases=False)
        scan.append(
            {
                "config": config_dict(config),
                "eligible": eligible(summary),
                "summary": summary,
            }
        )
        print(f"[{index:02d}/{len(grid()):02d}] scale grid", flush=True)
    qualified = [row for row in scan if row["eligible"]]
    pool = qualified or scan
    selected = min(
        pool,
        key=lambda row: (
            row["summary"]["candidate"]["joint_error_m"]
            + row["summary"]["candidate"]["vertex_error_m"],
            row["summary"]["candidate"]["pelvis_joint_error_m"]
            + row["summary"]["candidate"]["pelvis_vertex_error_m"],
            row["summary"]["joint_person_harm_over_5cm_rate"],
            row["summary"]["vertex_person_harm_over_5cm_rate"],
            json.dumps(row["config"], sort_keys=True),
        ),
    )
    report = {
        "experiment": "v14_brtc_person_local_robust_body_scale",
        "phase": "development",
        "protocol": {
            "data_read": "three offset0 only",
            "observable": "same matched-person stable predicted pre/post bone lengths",
            "gt_use": "parameter selection metrics only; never runtime inference",
            "action": "scale joints+vertices around bit-exact native root",
            "unchanged": "camera, root, pair-root layout, rejected, unmatched",
        },
        "grid_count": len(scan),
        "eligible_count": len(qualified),
        "selection_rule": (
            "joint, vertex, pelvis-centered joint/vertex improve or preserve; root/pair exact; "
            "person joint/vertex >5cm harm <=5%; minimize joint+vertex then centered errors"
        ),
        "selected": selected,
        "scan": scan,
        "dev_pass": bool(qualified),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "DEV_SCAN.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "DEV_SCAN.md").write_text(dev_markdown(report), encoding="utf-8")
    print(dev_markdown(report), flush=True)


def run_freeze(args: argparse.Namespace) -> None:
    dev_path = args.output_dir / "DEV_SCAN.json"
    report = json.loads(dev_path.read_text(encoding="utf-8"))
    if not bool(report["dev_pass"]):
        raise RuntimeError("No eligible development body-scale policy")
    frozen = {
        "experiment": "v14_brtc_person_local_robust_body_scale",
        "phase": "frozen_before_heldout",
        "frozen": True,
        "development_source": str(dev_path),
        "development_sha256": sha256(dev_path),
        "policy": report["selected"]["config"],
        "development_summary": report["selected"]["summary"],
        "contract": {
            "input": "last-pre/current-post same-person Human3R joints after frozen BRTC",
            "pivot": "native post root, bit-exact",
            "output": "root unchanged; joints and vertices scaled only",
            "fallback": "BRTC rejected, unmatched, insufficient edges, or high MAD exact v1/B0",
            "camera": "no update",
            "future_image_gt_model": "none",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(
        json.dumps(jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy": str(args.policy),
                "sha256": sha256(args.policy),
                "config": frozen["policy"],
            },
            indent=2,
        ),
        flush=True,
    )


def post_shot_acceleration(
    records: list[dict[str, Any]], config: BodyScaleConfig
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for record in records:
        case = record["case"]["case"]
        key = (
            record["case"]["sequence"],
            int(case["timestamp"]),
            int(case["source_camera"]),
            int(case["target_camera"]),
        )
        groups.setdefault(key, {})[int(case["offset"])] = record
    candidate_errors = {"root": [], "joint": [], "vertex": []}
    current_errors = {"root": [], "joint": [], "vertex": []}
    trajectories = 0
    for offsets in groups.values():
        if not all(index in offsets for index in (0, 1, 2)):
            continue
        boundary = offsets[0]
        first_scaled, debug = refine_brtc_output_body_scale(
            boundary["pre_people"],
            boundary["brtc_people"],
            boundary["accepted_matches"],
            config,
        )
        scale_by_identity = {}
        for post_index, person in enumerate(boundary["case"]["people"]):
            base = boundary["brtc_people"][post_index]
            scaled = first_scaled[post_index]
            base_extent = np.linalg.norm(base["joints"] - base["root"], axis=1).mean()
            scaled_extent = np.linalg.norm(scaled["joints"] - scaled["root"], axis=1).mean()
            scale_by_identity[person["identity"]] = float(
                scaled_extent / max(base_extent, 1e-12)
            )
        for identity, scale in scale_by_identity.items():
            candidate_series = {key: [] for key in ("root", "joints", "vertices")}
            current_series = {key: [] for key in ("root", "joints", "vertices")}
            targets = {key: [] for key in ("root", "joints", "vertices")}
            valid = True
            # Use the frozen boundary BRTC shift for the post shot, then apply
            # one frozen person scale at each frame's own bit-exact root.
            boundary_index = {
                person["identity"]: index
                for index, person in enumerate(boundary["case"]["people"])
            }[identity]
            boundary_shift = (
                np.asarray(boundary["brtc_people"][boundary_index]["root"])
                - np.asarray(boundary["case"]["people"][boundary_index]["post"]["root"])
            )
            for offset in (0, 1, 2):
                row = offsets[offset]
                lookup = {
                    person["identity"]: index
                    for index, person in enumerate(row["case"]["people"])
                }
                if identity not in lookup:
                    valid = False
                    break
                index = lookup[identity]
                native = row["case"]["people"][index]["post"]
                translated = {
                    key: np.asarray(native[key]) + boundary_shift
                    for key in ("root", "joints", "vertices")
                }
                scaled = scale_person_about_root(translated, scale)
                for key in ("root", "joints", "vertices"):
                    candidate_series[key].append(np.asarray(scaled[key]))
                    current_series[key].append(np.asarray(translated[key]))
                    targets[key].append(
                        np.asarray(row["case"]["people"][index]["target"][key])
                    )
            if not valid:
                continue
            trajectories += 1
            for key, metric_key in (
                ("root", "root"),
                ("joints", "joint"),
                ("vertices", "vertex"),
            ):
                source_rows = candidate_series[key]
                base_rows = current_series[key]
                target_rows = targets[key]
                if np.asarray(source_rows[0]).ndim > 1:
                    count = min(
                        *(len(value) for value in source_rows),
                        *(len(value) for value in base_rows),
                        *(len(value) for value in target_rows),
                    )
                    source_rows = [value[:count] for value in source_rows]
                    base_rows = [value[:count] for value in base_rows]
                    target_rows = [value[:count] for value in target_rows]
                target_delta2 = target_rows[2] - 2 * target_rows[1] + target_rows[0]
                for rows, destination in (
                    (source_rows, candidate_errors[metric_key]),
                    (base_rows, current_errors[metric_key]),
                ):
                    delta2 = rows[2] - 2 * rows[1] + rows[0]
                    destination.append(
                        float(np.linalg.norm(delta2 - target_delta2, axis=-1).mean())
                    )
    output = {"trajectory_count": trajectories}
    for name, values in (("candidate", candidate_errors), ("brtc_v1", current_errors)):
        for key in ("root", "joint", "vertex"):
            output[f"{name}_{key}_accel_delta2_mm_per_frame2"] = (
                finite_mean(values[key]) * 1000.0
            )
    return output


def replay_ego_scale(
    base_chains: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    config: BodyScaleConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Propagate body scale over BRTC shots while preserving every BRTC root."""

    by_key = {
        (int(row["chain_index"]), int(row["cut_index"])): row
        for row in boundary_rows
    }
    output, runtime_rows = [], []
    for base_chain in base_chains:
        chain_index = int(base_chain["chain_index"])
        base_segments = base_chain["segments"]
        candidate_segments = [copy.deepcopy(base_segments[0])]
        for segment_index in (1, 2):
            post_frames = copy.deepcopy(base_segments[segment_index])
            pre_frame = candidate_segments[-1][-1]
            pre_people = list(pre_frame["people"])
            pre_index_by_track = {
                int(person["global_track_id"]): index
                for index, person in enumerate(pre_people)
            }
            frozen = by_key[(chain_index, segment_index - 1)]
            track_post_pairs = sorted(
                frozen["association"]["track_to_post_index"].items()
            )
            accepted_post = {
                int(row["post_index"])
                for row in frozen["brtc"]["people"]
                if bool(row["accepted"])
            }
            accepted_matches = [
                (pre_index_by_track[int(track)], int(post_index))
                for track, post_index in track_post_pairs
                if int(post_index) in accepted_post
            ]
            before = post_frames[0]["people"]
            corrected, debug = refine_brtc_output_body_scale(
                pre_people, before, accepted_matches, config
            )
            debug_by_post = {
                int(row["post_index"]): row for row in debug["people"]
            }
            scale_by_native = {
                int(before[index]["native_track_id"]): float(row["scale_factor"])
                for index, row in debug_by_post.items()
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
            for expected, actual in zip(corrected, post_frames[0]["people"]):
                for key in ("root", "joints", "vertices"):
                    first_delta = max(
                        first_delta,
                        float(np.max(np.abs(np.asarray(expected[key]) - np.asarray(actual[key])))),
                    )
            runtime_rows.append(
                {
                    "chain_index": chain_index,
                    "cut_index": segment_index - 1,
                    "body_scale": debug,
                    "scale_by_native_track": scale_by_native,
                    "first_frame_replay_max_abs_delta": first_delta,
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
    return output, runtime_rows


def ego_person_harm(
    current_arrays: list[dict[str, np.ndarray]],
    candidate_arrays: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    output = {}
    for label, key in (
        ("fixed_joint", "fixed_joint_m"),
        ("fixed_vertex", "fixed_vertex_m"),
        ("pelvis_mpjpe", "pelvis_mpjpe_m"),
        ("pelvis_mpvpe", "pelvis_mpvpe_m"),
    ):
        old = np.concatenate([row[key] for row in current_arrays])
        new = np.concatenate([row[key] for row in candidate_arrays])
        delta = new - old
        output[label] = {
            "count": len(delta),
            "mean_delta_mm": float(delta.mean() * 1000.0),
            "improve_rate": float(np.mean(delta < 0.0)),
            "harm_over_1cm_rate": float(np.mean(delta > 0.01)),
            "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        }
    return output


def evaluate_ego(config: BodyScaleConfig, cache_path: Path) -> dict[str, Any]:
    from versions.v14 import eval_brtc_multithumbs_egohumans as ego

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    methods, boundary_rows = ego.method_chains(cache)
    candidate_chains, runtime_rows = replay_ego_scale(
        methods["b0_brtc_lc"], boundary_rows, config
    )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    reports, arrays_by_method, roots = {}, {}, {}
    for name, chains in (
        ("b0_brtc_lc", methods["b0_brtc_lc"]),
        ("b0_brtc_person_body_scale", candidate_chains),
    ):
        per_chain, arrays, root_rows = [], [], []
        for chain in chains:
            result, raw_arrays, root_errors = ego.evaluate_chain(
                chain, ego.DEFAULT_DATA, exo, vertex_map, joint_regressor, 30.0
            )
            per_chain.append(result)
            arrays.append(raw_arrays)
            root_rows.append(root_errors)
        reports[name] = ego.aggregate_method(per_chain, arrays)
        arrays_by_method[name] = arrays
        roots[name] = root_rows
    return {
        "geometry_cache": str(cache_path),
        "methods": reports,
        "person_harm_vs_brtc_v1": ego_person_harm(
            arrays_by_method["b0_brtc_lc"],
            arrays_by_method["b0_brtc_person_body_scale"],
        ),
        "root_harm_vs_brtc_v1": ego.harm_audit(
            roots["b0_brtc_lc"], roots["b0_brtc_person_body_scale"]
        ),
        "camera_exactness": ego.camera_exactness_audit(
            methods["b0_brtc_lc"], candidate_chains
        ),
        "runtime": {
            "boundary_count": len(runtime_rows),
            "first_frame_replay_max_abs_delta": max(
                row["first_frame_replay_max_abs_delta"] for row in runtime_rows
            ),
            "root_max_abs_change": max(
                row["body_scale"]["root_max_abs_change"] for row in runtime_rows
            ),
            "nonidentity_scale_count": sum(
                abs(scale - 1.0) > 1e-12
                for row in runtime_rows
                for scale in row["scale_by_native_track"].values()
            ),
            "scale_count": sum(
                len(row["scale_by_native_track"]) for row in runtime_rows
            ),
            "scales": [
                scale
                for row in runtime_rows
                for scale in row["scale_by_native_track"].values()
            ],
        },
        "boundary_rows": runtime_rows,
    }


def validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen BRTC person-local robust body-scale: blind validation",
        "",
        f"Frozen policy SHA256: `{report['policy_sha256_before_heldout']}`.",
        "",
    ]
    for name in ("three_offset1", "dance", "box"):
        lines.extend(table(name, report["splits"][name]["summary"]))
        accel = report["splits"][name].get("post_shot_acceleration")
        if accel:
            lines.append(f"Post-shot Accel: `{accel}`.\n")
    if "egohumans" in report:
        methods = report["egohumans"]["methods"]
        lines.extend(
            [
                "## EgoHumans same-forward CPU",
                "",
                "| Method | W | WA | Pelvis MPJPE | Pelvis MPVPE | Fixed joint | Fixed vertex | Joint Accel | Root | Pair dist | Pair vec |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("b0_brtc_lc", "b0_brtc_person_body_scale"):
            value = methods[name]["metrics"]
            lines.append(
                f"| {name} | {value['w_mpjpe_mm']:.3f} | {value['wa_mpjpe_mm']:.3f} | "
                f"{value['pelvis_mpjpe_mm']:.3f} | {value['pelvis_mpvpe_mm']:.3f} | "
                f"{value['fixed_world_joint_mm']:.3f} | {value['fixed_world_vertex_mm']:.3f} | "
                f"{value['world_joint_accel_delta2_mm_per_frame2']:.3f} | "
                f"{value['fixed_world_root_mm']:.3f} | {value['pairwise_root_distance_mm']:.3f} | "
                f"{value['pairwise_root_vector_mm']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"- Person harm: `{report['egohumans']['person_harm_vs_brtc_v1']}`.",
                f"- Runtime: `{report['egohumans']['runtime']}`.",
                "",
            ]
        )
    lines.extend(
        [
            f"- Held-out winner: **{report['heldout_winner']}**.",
            f"- Decision: **{report['decision']}**.",
            "- Frozen bytes were not changed after any held-out result was opened.",
        ]
    )
    return "\n".join(lines) + "\n"


def heldout_split_pass(summary: dict[str, Any]) -> bool:
    delta = summary["delta_vs_brtc_v1"]
    return bool(
        delta["joint_error_m"] <= 1e-12
        and delta["vertex_error_m"] <= 1e-12
        and delta["pelvis_joint_error_m"] <= 1e-12
        and delta["pelvis_vertex_error_m"] <= 1e-12
        and abs(delta["root_error_m"]) <= 1e-12
        and abs(delta["pairwise_distance_error_m"]) <= 1e-12
        and abs(delta["pairwise_vector_error_m"]) <= 1e-12
        and summary["joint_person_harm_over_5cm_rate"] <= 0.05
        and summary["vertex_person_harm_over_5cm_rate"] <= 0.05
        and summary["root_max_abs_change_vs_brtc"] == 0.0
    )


def run_validate(args: argparse.Namespace) -> None:
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    if not bool(frozen.get("frozen")):
        raise RuntimeError("Frozen pre-heldout policy required")
    policy_sha = sha256(args.policy)
    config = config_from_dict(frozen["policy"])
    # Held-out files are first touched only after policy bytes are hashed.
    split_rows = {
        "three_offset1": harness.load_rows(
            "confirm", args.confirm_report, args.max_cases
        ),
        "dance": legacy.report_rows(("dance",))[: int(args.max_cases) or None],
        "box": legacy.report_rows(("box",))[: int(args.max_cases) or None],
    }
    splits = {}
    for name, rows in split_rows.items():
        records = prepare_records(rows)
        summary = evaluate_config(records, config, include_cases=True)
        splits[name] = {"summary": summary}
        if name in ("dance", "box"):
            splits[name]["post_shot_acceleration"] = post_shot_acceleration(
                records, config
            )
    report = {
        "experiment": "v14_brtc_person_local_robust_body_scale",
        "phase": "blind_heldout",
        "policy_source": str(args.policy),
        "policy_sha256_before_heldout": policy_sha,
        "config": config_dict(config),
        "splits": splits,
    }
    if not args.skip_ego:
        report["egohumans"] = evaluate_ego(config, args.ego_geometry_cache)
    split_pass = {
        name: heldout_split_pass(value["summary"])
        for name, value in splits.items()
    }
    ego_pass = True
    if "egohumans" in report:
        ego = report["egohumans"]
        old = ego["methods"]["b0_brtc_lc"]["metrics"]
        new = ego["methods"]["b0_brtc_person_body_scale"]["metrics"]
        required = (
            "w_mpjpe_mm",
            "wa_mpjpe_mm",
            "pelvis_mpjpe_mm",
            "pelvis_mpvpe_mm",
            "fixed_world_joint_mm",
            "fixed_world_vertex_mm",
            "world_joint_accel_delta2_mm_per_frame2",
        )
        harm = ego["person_harm_vs_brtc_v1"]
        ego_pass = bool(
            all(new[key] <= old[key] + 1e-12 for key in required)
            and harm["fixed_joint"]["harm_over_5cm_rate"] <= 0.05
            and harm["fixed_vertex"]["harm_over_5cm_rate"] <= 0.05
            and ego["camera_exactness"]["bit_exact"]
            and ego["runtime"]["root_max_abs_change"] == 0.0
        )
    report["split_pass"] = split_pass
    report["egohumans_pass"] = ego_pass
    report["heldout_winner"] = bool(all(split_pass.values()) and ego_pass)
    report["decision"] = "PROMOTE" if report["heldout_winner"] else "NO_GO_ARCHIVE"
    report["policy_sha256_after_heldout"] = sha256(args.policy)
    report["policy_unchanged"] = bool(
        report["policy_sha256_after_heldout"] == policy_sha
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "HELDOUT_RESULTS.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = validation_markdown(report)
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(text, encoding="utf-8")
    doc = REPO_ROOT / "versions/v14/docs/V14_BRTC_PERSON_BODY_SCALE_20260801.md"
    doc.write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must remain in Movie3R /data workspace")
    if args.phase == "dev":
        run_dev(args)
    elif args.phase == "freeze":
        run_freeze(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()
