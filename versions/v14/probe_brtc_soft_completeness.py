#!/usr/bin/env python3
"""Freeze and confirm a soft incomplete-association BRTC action scale.

Linear completeness weighting is safer but over-damps person depth on the
independent MultiHuman variable-visibility split.  This probe learns one
bounded scalar only for incomplete rectangular associations.  Complete
one-to-one boundaries remain bit-exact frozen BRTC-LC v1::

    scale = 1                                if M == max(Npre, Npost)
            frozen_partial_scale             otherwise

Development uses the 22 previously excluded ``three`` cuts.  The scalar is
serialized before ``dance`` and ``box`` confirmation.  The candidate remains
strictly online, camera-frozen, and model-free beyond already-frozen B0.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.probe_b0_identity_matching import strict_cache  # noqa: E402
from versions.v14.probe_b0_two_view_person_triangulation import (  # noqa: E402
    point_errors,
    transform_points,
)
from versions.v14.probe_brtc_variable_visibility import (  # noqa: E402
    DEFAULT_SOURCE_REPORT,
    METHODS as UNUSED_METHODS,
    PRIMARY,
    anonymous_matches,
    finite_mean,
    jsonable,
    layout_errors,
    ordered_people,
    person_geometry,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402


DEFAULT_BOUNDARY_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_variable_visibility"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_soft_completeness"
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_CONFIRM.json"
DEFAULT_SCALES = (0.67, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
POINT_KEYS = ("root", "joints", "vertices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "freeze", "confirm"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--boundary_root", type=Path, default=DEFAULT_BOUNDARY_ROOT)
    parser.add_argument("--three_report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--dance_report",
        type=Path,
        default=REPO_ROOT / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
    )
    parser.add_argument(
        "--box_report",
        type=Path,
        default=REPO_ROOT / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
    )
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    return parser.parse_args()


def case_names(report_path: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return list(report["protocol"]["excluded_variable_visibility_cases"])


def scaled_people(
    post_people: list[dict[str, np.ndarray]],
    base_debug: dict[str, Any],
    partial_scale: float,
) -> tuple[list[dict[str, np.ndarray]], float]:
    denominator = max(
        int(base_debug.get("previous_observable_count", 0)),
        int(base_debug.get("current_observable_count", 0)),
    )
    matched = int(base_debug["matched_count"])
    scale = 1.0 if matched == denominator else float(partial_scale)
    corrected = [copy.deepcopy(person) for person in post_people]
    for record in base_debug["people"]:
        if not bool(record["accepted"]):
            continue
        shift = scale * np.asarray(record["final_shift_world"], dtype=np.float64)
        index = int(record["post_index"])
        for key in POINT_KEYS:
            corrected[index][key] = np.asarray(post_people[index][key], dtype=np.float64) + shift
    return corrected, scale


def evaluate_case(
    cache: dict[str, Any],
    boundary: np.ndarray,
    scales: tuple[float, ...],
) -> dict[str, Any]:
    pre_order = ordered_people(cache["humans"][-2])
    post_order = ordered_people(cache["humans"][-1])
    pre_people = [person_geometry(person) for _, person in pre_order]
    post_people = [person_geometry(person, boundary) for _, person in post_order]
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = boundary @ np.asarray(cache["poses"][-1], dtype=np.float64)
    matches, association = anonymous_matches(pre_order, post_order, boundary)
    base, debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    # Counts are supplied here rather than added to frozen v1 debug.
    debug = dict(debug)
    debug["previous_observable_count"] = len(pre_people)
    debug["current_observable_count"] = len(post_people)

    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    targets = [
        person_geometry(cache["gt"]["post_humans"][identity], gauge)
        for identity, _ in post_order
    ]
    predictions: dict[str, list[dict[str, np.ndarray]]] = {
        "b0": post_people,
        "v1": base,
    }
    applied_scales = {}
    for scale in scales:
        name = f"partial_{scale:.4f}"
        predictions[name], applied_scales[name] = scaled_people(post_people, debug, scale)

    base_root = np.asarray([
        point_errors(predicted, target)["root_error_m"]
        for predicted, target in zip(post_people, targets)
    ])
    metrics = {}
    for name, people in predictions.items():
        errors = [point_errors(predicted, target) for predicted, target in zip(people, targets)]
        value = {
            key: finite_mean([row[key] for row in errors])
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        } | layout_errors(people, targets)
        if name != "b0":
            roots = np.asarray([row["root_error_m"] for row in errors])
            delta = roots - base_root
            value.update({
                "root_improve_rate": float(np.mean(delta < 0.0)),
                "root_harm_over_5cm_rate": float(np.mean(delta > 0.05)),
                "root_mean_delta_m": float(np.mean(delta)),
            })
        metrics[name] = value
    return {
        "case": cache["case"],
        "pre_person_count": len(pre_people),
        "post_person_count": len(post_people),
        "association": association,
        "accepted_count": int(debug["accepted_count"]),
        "applied_scales": applied_scales,
        "metrics": metrics,
    }


def evaluate_sequence(
    sequence: str,
    report_path: Path,
    boundary_dir: Path,
    scales: tuple[float, ...],
) -> list[dict[str, Any]]:
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    args = argparse.Namespace(
        data_root=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
        size=512,
        sequence=sequence,
    )
    rows = []
    names = case_names(report_path)
    for index, name in enumerate(names, start=1):
        boundary_path = boundary_dir / "cases" / f"{name}.json"
        boundary_row = json.loads(boundary_path.read_text(encoding="utf-8"))
        cache = strict_cache(args, SEQUENCE_INPUTS[sequence]["cache"] / f"{name}.pt")
        rows.append(evaluate_case(
            cache,
            np.asarray(boundary_row["learned_b0"], dtype=np.float64),
            scales,
        ))
        print(f"[{sequence} {index:02d}/{len(names):02d}] {name}", flush=True)
    return rows


def aggregate(cases: list[dict[str, Any]], method: str) -> dict[str, float]:
    result = {
        key: finite_mean([row["metrics"][method][key] for row in cases])
        for key in PRIMARY
    }
    if method != "b0":
        result.update({
            key: finite_mean([row["metrics"][method][key] for row in cases])
            for key in ("root_improve_rate", "root_harm_over_5cm_rate", "root_mean_delta_m")
        })
    return result


def choose_policy(cases: list[dict[str, Any]], scales: tuple[float, ...]) -> dict[str, Any]:
    reference = aggregate(cases, "v1")
    scan = {}
    eligible = []
    for scale in scales:
        name = f"partial_{scale:.4f}"
        value = aggregate(cases, name)
        scan[name] = value
        if (
            all(value[key] <= reference[key] + 1e-12 for key in PRIMARY)
            and value["root_harm_over_5cm_rate"] <= reference["root_harm_over_5cm_rate"] + 1e-12
        ):
            eligible.append(float(scale))
    selected = min(
        eligible,
        key=lambda scale: (
            scan[f"partial_{scale:.4f}"]["root_harm_over_5cm_rate"],
            scan[f"partial_{scale:.4f}"]["root_error_m"],
            scale,
        ),
    ) if eligible else 1.0
    return {
        "partial_scale": float(selected),
        "eligible_scales": eligible,
        "selection_rule": (
            "all five primary means and harm no worse than frozen v1 on three variable; "
            "then minimum harm, root, scale"
        ),
        "reference_v1": reference,
        "scan": scan,
        "development_pass": bool(eligible and selected < 1.0),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Soft completeness BRTC",
        "",
        f"Phase: `{report['phase']}`; frozen partial scale: `{report['policy']['partial_scale']:.4f}`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, methods in report["results"].items():
        for name, value in methods.items():
            harm = "-" if name == "b0" else f"{value['root_harm_over_5cm_rate']:.1%}"
            lines.append(
                f"| {split} | {name} | {value['root_error_m']:.4f} | {value['joint_error_m']:.4f} | "
                f"{value['vertex_error_m']:.4f} | {value['pairwise_distance_error_m']:.4f} | "
                f"{value['pairwise_vector_error_m']:.4f} | {harm} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in Movie3R under /data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = tuple(sorted(set(float(value) for value in args.scales)))
    if 1.0 not in scales:
        raise ValueError("Scale grid must include 1.0")

    if args.phase == "dev":
        cases = evaluate_sequence("three", args.three_report, args.boundary_root, scales)
        policy = choose_policy(cases, scales)
        results = {
            "three_variable_dev": {
                "b0": aggregate(cases, "b0"),
                "v1": aggregate(cases, "v1"),
                "selected": aggregate(cases, f"partial_{policy['partial_scale']:.4f}"),
            }
        }
        report_name = "DEV_SCAN.json"
    elif args.phase == "freeze":
        dev_path = args.output_dir / "DEV_SCAN.json"
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        policy = dev["policy"]
        if not policy["development_pass"]:
            raise RuntimeError("No nontrivial soft-completeness policy passed development")
        frozen = {
            "experiment": "v14_soft_completeness_brtc",
            "frozen_before_confirmation": True,
            "policy": policy,
            "constraints": {
                "future_frames": 0,
                "extra_pretrained_models": [],
                "camera_update": "none",
                "complete_boundary": "bit-exact frozen BRTC-LC v1",
            },
        }
        args.policy.write_text(
            json.dumps(jsonable(frozen), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(args.policy)
        return
    else:
        frozen = json.loads(args.policy.read_text(encoding="utf-8"))
        policy = frozen["policy"]
        selected = float(policy["partial_scale"])
        evaluation_scales = tuple(sorted(set((selected, 1.0))))
        dance = evaluate_sequence(
            "dance", args.dance_report, args.boundary_root / "dance", evaluation_scales
        )
        box = evaluate_sequence(
            "box", args.box_report, args.boundary_root / "box", evaluation_scales
        )
        results = {}
        for name, cases in (("dance_variable_confirm", dance), ("box_variable_confirm", box)):
            results[name] = {
                "b0": aggregate(cases, "b0"),
                "v1": aggregate(cases, "v1"),
                "selected": aggregate(cases, f"partial_{selected:.4f}"),
            }
        report_name = "CONFIRM_RESULTS.json"

    report = {
        "experiment": "v14_soft_completeness_brtc",
        "phase": args.phase,
        "protocol": {
            "development": "three variable-visibility cuts excluded by prior identity work",
            "confirmation": "dance and box variable-visibility cuts after policy freeze",
            "candidate_gt_use": "development metric fitting only; none at inference",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
        },
        "policy": policy,
        "results": results,
    }
    (args.output_dir / report_name).write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / report_name.replace(".json", ".md")).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
