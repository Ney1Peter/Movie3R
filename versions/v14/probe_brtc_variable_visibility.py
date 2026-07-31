#!/usr/bin/env python3
"""Independent entry/exit confirmation for completeness-weighted BRTC.

The original B0 identity experiment excluded cuts whose visible person sets
changed across the boundary.  Those cuts are an evaluation-disjoint test of
the association-completeness rule discovered on EgoHumans.  This probe
recovers frozen-B0 boundaries, performs anonymous rectangular Hungarian
association, and compares frozen BRTC-LC v1 with the completeness wrapper.

Candidate prediction uses no GT, future frame, or extra pretrained model.
GT-labelled cache fields are read only after prediction for metrics and for
the identity-correctness audit.  Cameras are never modified by either person
refinement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_identity_matching as identity_probe  # noqa: E402
from versions.v14.b0_person_triangulation import (  # noqa: E402
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_completeness_weighted import (  # noqa: E402
    refine_matched_people_completeness_weighted,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_two_view_person_triangulation import (  # noqa: E402
    point_errors,
    transform_points,
)
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth"
DEFAULT_SOURCE_REPORT = REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json"
DEFAULT_SOURCE_CASES = REPO_ROOT / "output/v14/b0_identity_matching/cases"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_variable_visibility"
POINT_KEYS = ("root", "joints", "vertices")
METHODS = ("b0", "b0_brtc_lc", "b0_brtc_completeness_weighted")
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source_report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--source_case_dir", type=Path, default=DEFAULT_SOURCE_CASES)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument(
        "--sequence", choices=tuple(geometry.SEQUENCE_IDENTITIES), default="three"
    )
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--evaluation_only", action="store_true")
    parser.add_argument("--overwrite_boundaries", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def ordered_people(frame: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        ((str(identity), person) for identity, person in frame.items()),
        key=lambda row: int(row[1]["detection_index"]),
    )


def person_geometry(person: dict[str, Any], transform: np.ndarray | None = None) -> dict[str, np.ndarray]:
    output = {}
    for key in POINT_KEYS:
        points = np.asarray(person[key], dtype=np.float64)
        output[key] = transform_points(transform, points) if transform is not None else points.copy()
    return output


def anonymous_matches(
    pre_order: list[tuple[str, dict[str, Any]]],
    post_order: list[tuple[str, dict[str, Any]]],
    boundary: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pre_by_identity = {identity: person for identity, person in pre_order}
    pre_keys = tuple(identity for identity, _ in pre_order)
    components = identity_probe.identity_cost_components(
        pre_by_identity,
        post_order,
        boundary,
        pre_keys,
    )
    cost = identity_probe.matching_costs(components)["root_torso_joints"]
    rows, columns = linear_sum_assignment(cost)
    matches = [(int(row), int(column)) for row, column in zip(rows, columns)]
    correctness = [pre_order[row][0] == post_order[column][0] for row, column in matches]
    return matches, {
        "matcher": "anonymous rectangular Hungarian root+torso+joints",
        "cost_shape": tuple(int(value) for value in cost.shape),
        "matches": matches,
        "pre_evaluator_id_by_index": [identity for identity, _ in pre_order],
        "post_evaluator_id_by_index": [identity for identity, _ in post_order],
        "correct_count_evaluator_only": int(sum(correctness)),
        "matched_count": len(matches),
        "accuracy_evaluator_only": float(np.mean(correctness)) if correctness else float("nan"),
        "cost": cost,
    }


def layout_errors(
    people: list[dict[str, np.ndarray]],
    targets: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    distance_errors, vector_errors = [], []
    for first in range(len(people)):
        for second in range(first + 1, len(people)):
            predicted = people[first]["root"] - people[second]["root"]
            target = targets[first]["root"] - targets[second]["root"]
            distance_errors.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
            vector_errors.append(float(np.linalg.norm(predicted - target)))
    return {
        "pairwise_distance_error_m": finite_mean(distance_errors),
        "pairwise_vector_error_m": finite_mean(vector_errors),
    }


def evaluate_case(cache: dict[str, Any], boundary: np.ndarray) -> dict[str, Any]:
    pre_order = ordered_people(cache["humans"][-2])
    post_order = ordered_people(cache["humans"][-1])
    pre_people = [person_geometry(person) for _, person in pre_order]
    post_people = [person_geometry(person, boundary) for _, person in post_order]

    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    raw_post_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    post_camera = np.asarray(boundary, dtype=np.float64) @ raw_post_camera
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post

    matches, association = anonymous_matches(pre_order, post_order, boundary)
    corrected_v1, debug_v1 = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    corrected_weighted, debug_weighted = refine_matched_people_completeness_weighted(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    predictions = {
        "b0": post_people,
        "b0_brtc_lc": corrected_v1,
        "b0_brtc_completeness_weighted": corrected_weighted,
    }

    targets = []
    for identity, _ in post_order:
        if identity not in cache["gt"]["post_humans"]:
            raise KeyError(f"Detected post identity {identity} has no evaluator target")
        targets.append(person_geometry(cache["gt"]["post_humans"][identity], gauge))

    per_method = {}
    person_rows = []
    for method in METHODS:
        errors = [point_errors(predicted, target) for predicted, target in zip(predictions[method], targets)]
        layout = layout_errors(predictions[method], targets)
        per_method[method] = {
            key: finite_mean([row[key] for row in errors])
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        } | layout
        person_rows.append(errors)

    base_person_errors = [
        point_errors(predicted, target)["root_error_m"]
        for predicted, target in zip(predictions["b0"], targets)
    ]
    harm = {}
    for method in METHODS[1:]:
        method_errors = [
            point_errors(predicted, target)["root_error_m"]
            for predicted, target in zip(predictions[method], targets)
        ]
        deltas = np.asarray(method_errors) - np.asarray(base_person_errors)
        harm[method] = {
            "root_improve_rate": float(np.mean(deltas < 0.0)),
            "root_harm_over_1cm_rate": float(np.mean(deltas > 0.01)),
            "root_harm_over_5cm_rate": float(np.mean(deltas > 0.05)),
            "root_mean_delta_m": float(np.mean(deltas)),
        }

    return {
        "case": cache["case"],
        "pre_person_count": len(pre_people),
        "post_person_count": len(post_people),
        "association": association,
        "camera": {
            "b0_translation_error_m": float(np.linalg.norm(post_camera[:3, 3] - target_camera[:3, 3])),
            "candidate_max_abs_change": 0.0,
        },
        "runtime": {
            "v1_matched_count": int(debug_v1["matched_count"]),
            "v1_accepted_count": int(debug_v1["accepted_count"]),
            "completeness": float(debug_weighted["completeness"]),
            "action_scale": float(debug_weighted["action_scale"]),
        },
        "methods": per_method,
        "harm_vs_b0": harm,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "case_count": len(cases),
        "post_person_count": int(sum(row["post_person_count"] for row in cases)),
        "matched_count": int(sum(row["association"]["matched_count"] for row in cases)),
        "accepted_count": int(sum(row["runtime"]["v1_accepted_count"] for row in cases)),
        "association_accuracy_evaluator_only": finite_mean(
            [row["association"]["accuracy_evaluator_only"] for row in cases]
        ),
        "camera_candidate_max_abs_change": float(max(
            (row["camera"]["candidate_max_abs_change"] for row in cases), default=0.0
        )),
        "population_transitions": {},
        "methods": {},
    }
    for row in cases:
        key = f"{row['pre_person_count']}->{row['post_person_count']}"
        output["population_transitions"][key] = output["population_transitions"].get(key, 0) + 1
    for method in METHODS:
        output["methods"][method] = {
            metric: finite_mean([row["methods"][method][metric] for row in cases])
            for metric in PRIMARY
        }
        if method != "b0":
            for metric in (
                "root_improve_rate",
                "root_harm_over_1cm_rate",
                "root_harm_over_5cm_rate",
                "root_mean_delta_m",
            ):
                output["methods"][method][metric] = finite_mean(
                    [row["harm_vs_b0"][method][metric] for row in cases]
                )
    reference = output["methods"]["b0_brtc_lc"]
    candidate = output["methods"]["b0_brtc_completeness_weighted"]
    output["candidate_vs_v1_delta"] = {
        key: float(candidate[key] - reference[key])
        for key in PRIMARY + ("root_harm_over_5cm_rate",)
    }
    output["strict_all_primary_non_regression"] = bool(
        all(candidate[key] <= reference[key] + 1e-12 for key in PRIMARY)
    )
    output["candidate_winner"] = bool(
        output["strict_all_primary_non_regression"]
        and candidate["root_harm_over_5cm_rate"] <= reference["root_harm_over_5cm_rate"] + 1e-12
        and any(candidate[key] < reference[key] - 1e-12 for key in PRIMARY)
        and output["camera_candidate_max_abs_change"] <= 1e-12
    )
    return output


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# BRTC variable-visibility independent confirmation",
        "",
        f"Cases: `{summary['case_count']}`; transitions: `{summary['population_transitions']}`.",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = summary["methods"][method]
        harm = "-" if method == "b0" else f"{value['root_harm_over_5cm_rate']:.1%}"
        lines.append(
            f"| {method} | {value['root_error_m']:.4f} | {value['joint_error_m']:.4f} | "
            f"{value['vertex_error_m']:.4f} | {value['pairwise_distance_error_m']:.4f} | "
            f"{value['pairwise_vector_error_m']:.4f} | {harm} |"
        )
    lines.extend([
        "",
        f"Strict all-primary non-regression: `{summary['strict_all_primary_non_regression']}`.",
        f"Candidate winner: `{summary['candidate_winner']}`.",
        "Camera candidate max change: "
        f"`{summary['camera_candidate_max_abs_change']:.3e}`.",
        "",
        "This split was excluded by the earlier same-visibility B0 identity experiment and was not used "
        "to discover the completeness formula.",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> None:
    pre = [
        ("a", {"detection_index": 1, "root": np.array([0.0, 0.0, 2.0]), "torso": np.eye(3), "joints": np.zeros((3, 3))}),
        ("b", {"detection_index": 2, "root": np.array([1.0, 0.0, 2.0]), "torso": np.eye(3), "joints": np.ones((3, 3))}),
    ]
    post = [pre[1]]
    matches, audit = anonymous_matches(pre, post, np.eye(4))
    assert matches == [(1, 0)]
    assert audit["cost_shape"] == (2, 1)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in Movie3R under /data")
    if args.cache_dir is None:
        args.cache_dir = SEQUENCE_INPUTS[str(args.sequence)]["cache"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    case_names = list(source["protocol"]["excluded_variable_visibility_cases"])
    if args.max_cases > 0:
        case_names = case_names[: args.max_cases]

    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[str(args.sequence)]
    model = None
    model_flags = None
    boundary_rows = []
    for index, case_name in enumerate(case_names, start=1):
        boundary_path = case_dir / f"{case_name}.json"
        original_path = args.source_case_dir / f"{case_name}.json"
        if boundary_path.is_file() and not args.overwrite_boundaries:
            boundary_row = json.loads(boundary_path.read_text(encoding="utf-8"))
        elif original_path.is_file() and not args.overwrite_boundaries:
            original = json.loads(original_path.read_text(encoding="utf-8"))
            boundary_row = {
                "case": original["case"],
                "learned_b0": original["boundaries"]["learned_b0"],
                "source": str(original_path),
            }
        else:
            if args.evaluation_only:
                raise FileNotFoundError(f"Missing B0 boundary: {boundary_path}")
            if model is None:
                if not args.model_path.is_file():
                    raise FileNotFoundError(args.model_path)
                from dust3r.model import ARCroco3DStereo

                model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
                model_flags = configure_model(model)
            cache = identity_probe.strict_cache(args, args.cache_dir / f"{case_name}.pt")
            learned_b0, inference = identity_probe.infer_learned_b0(model, args, cache)
            boundary_row = {
                "case": cache["case"],
                "learned_b0": learned_b0,
                "inference": inference,
                "source": "fresh frozen-B0 inference",
            }
        boundary_path.write_text(
            json.dumps(jsonable(boundary_row), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        boundary_rows.append(boundary_row)
        print(f"[{index:02d}/{len(case_names):02d}] boundary {case_name}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cases = []
    for index, (case_name, boundary_row) in enumerate(zip(case_names, boundary_rows), start=1):
        cache = identity_probe.strict_cache(args, args.cache_dir / f"{case_name}.pt")
        case = evaluate_case(cache, np.asarray(boundary_row["learned_b0"], dtype=np.float64))
        cases.append(case)
        print(
            f"[{index:02d}/{len(case_names):02d}] evaluate {case_name} "
            f"{case['pre_person_count']}->{case['post_person_count']} "
            f"scale={case['runtime']['action_scale']:.3f}",
            flush=True,
        )
        del cache

    report = {
        "experiment": "v14_brtc_variable_visibility_independent_confirmation",
        "protocol": {
            "sequence": f"MultiHuman {args.sequence} variable-visibility cuts",
            "split_source": str(args.source_report),
            "selection": "all cuts excluded earlier because pre/post visible sets differ",
            "candidate_discovery_data": "EgoHumans only; this split was unseen by the completeness rule",
            "association": "anonymous rectangular Hungarian using frozen-B0 root+torso+joints",
            "candidate_gt_use": "none; evaluator only",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "checkpoint": str(args.model_path),
            "model_flags": model_flags,
        },
        "summary": summarize(cases),
        "cases": cases,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
