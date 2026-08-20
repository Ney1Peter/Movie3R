#!/usr/bin/env python3
"""GT-only EgoHumans evaluator using the frozen v15 metric implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import (  # noqa: E402
    MIN_VISIBLE_VERTEX_FRACTION,
    detector_metrics,
    evaluate_method,
    method_arrays,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v19.egohumans.dataset import (  # noqa: E402
    jsonable,
    load_exo_calibrations,
    load_smpl_people,
    visibility_fraction,
)


def load_gt(
    record: dict[str, Any],
    extracted_root: Path,
    topology: CommonTopology,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load EgoHumans SMPL and cameras in the metric Aria world."""

    capture_root = extracted_root.resolve() / str(record["capture_relative"])
    calibrations = load_exo_calibrations(capture_root)
    frames = [int(value) for value in record["pre_frame_numbers"] + record["post_frame_numbers"]]
    cameras = (
        [str(record["pre_camera"])] * len(record["pre_frame_numbers"])
        + [str(record["post_camera"])] * len(record["post_frame_numbers"])
    )
    missing = sorted(set(cameras) - set(calibrations))
    if missing:
        raise KeyError(f"Missing exo calibration: {missing}")
    first = load_smpl_people(capture_root, frames[0])
    identities = sorted(first)
    vertices, joints, fractions = [], [], []
    for frame, camera in zip(frames, cameras):
        people = load_smpl_people(capture_root, frame)
        if sorted(people) != identities:
            raise ValueError(
                f"GT identity set changes at frame {frame}: {sorted(people)} vs {identities}"
            )
        frame_vertices = np.stack([people[identity]["vertices"] for identity in identities])
        vertices.append(frame_vertices)
        joints.append(topology.joints_from_smpl(frame_vertices))
        fractions.append(
            [
                visibility_fraction(
                    capture_root,
                    camera,
                    frame,
                    identity,
                    people[identity]["vertices"],
                    calibrations[camera],
                )
                for identity in identities
            ]
        )
    visible_fraction = np.asarray(fractions, dtype=np.float64)
    return (
        {
            "cameras_c2w": np.stack(
                [calibrations[camera].camera_to_world for camera in cameras]
            ),
            "vertices_world": np.stack(vertices),
            "joints_world": np.stack(joints),
            "frames": np.asarray(frames, dtype=np.int64),
            "visible_fraction": visible_fraction,
            "visible": visible_fraction >= MIN_VISIBLE_VERTEX_FRACTION,
        },
        identities,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = runtime["record"]
    topology = CommonTopology.load()
    gt, identities = load_gt(record, args.extracted_root, topology)
    results = {}
    errors = {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in runtime["methods"]:
            try:
                results[method] = evaluate_method(
                    method,
                    method_arrays(cache, method),
                    gt,
                    identities,
                    int(record["boundary_index"]),
                    float(record["fps"]),
                )
            except Exception as error:  # preserve every failed method in the audit
                errors[method] = f"{type(error).__name__}: {error}"
    detectors = {}
    for name, key in {
        "causal_gru": "causal_gru_detector",
        "static_logistic": "static_logistic_detector",
    }.items():
        if key in runtime.get("runtime", {}):
            detectors[name] = detector_metrics(runtime, int(record["boundary_index"]), key)
    payload = {
        "schema_version": "Movie3R-v19-EgoHumans-evaluation-v1",
        "protocol": record["protocol"],
        "case_id": record["case_id"],
        "record": record,
        "identities": identities,
        "methods": results,
        "errors": errors,
        "detectors": detectors,
        "gt_summary": {
            "coordinate_system": "metric Aria world",
            "body_model": "SMPL-6890",
            "visible_gt_person_frames": int(gt["visible"].sum()),
            "frame_count": len(gt["frames"]),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "runtime_gt_access": False,
            "common_topology": topology.metadata(),
            "matching": "per-frame Hungarian in camera coordinates; GT identity never enters runtime",
            "literature_protocol_status": (
                "Multi-THuMBS public-description reproduction; exact official capture/cut/evaluator unavailable"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(args.output)
    concise = {
        method: {
            "W": value["multi_thumbs_named_provisional"]["w_mpjpe_mm"]["mean"],
            "WA": value["multi_thumbs_named_provisional"]["wa_mpjpe_mm"]["mean"],
            "MPJPE": value["multi_thumbs_named_provisional"]["mpjpe_mm"]["mean"],
            "MPVPE": value["multi_thumbs_named_provisional"]["mpvpe_mm"]["mean"],
            "Accel": value["multi_thumbs_named_provisional"]["accel_delta2_mm_per_frame2"]["mean"],
            "ATE": value["multi_thumbs_named_provisional"]["ate_sim3_m"]["mean"],
            "IDs": value["identity"]["ids_total"],
            "IDF1": value["identity"]["idf1"],
            "Coverage": value["coverage"]["coverage"],
        }
        for method, value in results.items()
    }
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "case_id": record["case_id"],
                "metrics": concise,
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

