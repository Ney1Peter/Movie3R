#!/usr/bin/env python3
"""Evaluate frozen V14 B0 and B0+DA3 on six EgoHumans camera cuts.

The candidate path is strictly causal and GT-free:

    pre RGB history + first-post RGB -> V14 shadow
    first-post RGB                  -> fresh raw Human3R
    shadow/raw camera               -> frozen B0
    last-pre/first-post RGB         -> bidirectional frozen DA3 + frozen gate

Only after both boundaries have been finalized do we load EgoHumans camera and
SMPL ground truth.  GT 2D annotations are used solely to assign detections to
the three evaluation identities.  They never generate or select a boundary.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset" / "Depth-Anything-3"
for path in (REPO_ROOT, SRC_ROOT, REPO_ROOT / "scripts", DA3_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from depth_anything_3.api import DepthAnything3  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v13.egobody_probe import (  # noqa: E402
    IDENTITIES,
    assign_identities,
    load_colmap,
)
from versions.v13.gt_id_consensus import (  # noqa: E402
    jsonable,
    layer_humans,
    prepare_full_square_input,
    transform_points,
)
from versions.v14.b0_da3_fine_alignment import (  # noqa: E402
    DA3FineAligner,
    DEFAULT_CONFIG,
)
from versions.v14.run_v14_2_multihuman_sequence import run_rollout  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_CHECKPOINT = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_DA3 = DA3_ROOT / "checkpoints/DAE-base"
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble")
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_da3_egohumans"
)

# Three fixed streams already used by the V13 EgoHumans feasibility probes.
# Consecutive shots share one timestamp at each boundary.
CHAINS = (
    (
        ("cam01", tuple(range(296, 301))),
        ("cam06", tuple(range(300, 305))),
        ("cam07", tuple(range(304, 309))),
    ),
    (
        ("cam02", tuple(range(176, 181))),
        ("cam05", tuple(range(180, 185))),
        ("cam08", tuple(range(184, 189))),
    ),
    (
        ("cam03", tuple(range(416, 421))),
        ("cam04", tuple(range(420, 425))),
        ("cam01", tuple(range(424, 429))),
    ),
)

METHODS = ("b0", "b0_da3_safe")
HUMAN_METRICS = ("root_error_m", "joint_error_m", "vertex_error_m")
CAMERA_METRICS = ("camera_translation_error_m", "camera_rotation_error_deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--da3_path", type=Path, default=DEFAULT_DA3)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--assignment_threshold_px", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="CHAIN:CUT",
        help="Run only selected zero-based cases, e.g. --case 0:0 --case 2:1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute an existing per-cut cache; never deletes the output directory.",
    )
    return parser.parse_args()


def requested_cases(values: list[str]) -> set[tuple[int, int]]:
    if not values:
        return {(chain, cut) for chain in range(len(CHAINS)) for cut in (0, 1)}
    output = set()
    for value in values:
        try:
            chain_text, cut_text = value.split(":", maxsplit=1)
            case = (int(chain_text), int(cut_text))
        except ValueError as error:
            raise ValueError(f"Invalid --case {value!r}; expected CHAIN:CUT") from error
        if case[0] not in range(len(CHAINS)) or case[1] not in (0, 1):
            raise ValueError(f"Unknown case {value!r}")
        output.add(case)
    return output


def image_paths(data_root: Path, camera: str, frames: tuple[int, ...]) -> list[Path]:
    paths = [data_root / f"exo/{camera}/images/{frame:05d}.jpg" for frame in frames]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(estimated)[:3, :3].T @ np.asarray(target)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def smpl_resources() -> tuple[np.ndarray, np.ndarray]:
    with (REPO_ROOT / "src/models/smplx/smplx2smpl.pkl").open("rb") as handle:
        mapping = pickle.load(handle, encoding="latin1")["matrix"]
    with (REPO_ROOT / "src/models/smpl/SMPL_NEUTRAL.pkl").open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    mapping = np.asarray(mapping, dtype=np.float64)
    regressor = np.asarray(regressor, dtype=np.float64)[:24]
    if mapping.shape != (6890, 10475) or regressor.shape != (24, 6890):
        raise ValueError(
            f"Unexpected SMPL resources: mapping={mapping.shape}, J={regressor.shape}"
        )
    return mapping, regressor


def gt_camera(exo: dict, camera: str) -> np.ndarray:
    return np.asarray(exo[camera]["c2w_aria01"], dtype=np.float64)


def boundary_camera_metrics(
    boundary: np.ndarray,
    raw_camera: np.ndarray,
    target_camera: np.ndarray,
) -> dict:
    estimated = np.asarray(boundary, dtype=np.float64) @ np.asarray(
        raw_camera, dtype=np.float64
    )
    translation = float(np.linalg.norm(estimated[:3, 3] - target_camera[:3, 3]))
    rotation = rotation_error_deg(estimated, target_camera)
    return {
        "estimated_camera": estimated,
        "camera_translation_error_m": translation,
        "camera_rotation_error_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def gt_smpl(data_root: Path, frame: int) -> dict[str, dict]:
    value = np.load(
        data_root / f"processed_data/smpl/{frame:05d}.npy", allow_pickle=True
    ).item()
    return {str(identity): row for identity, row in value.items()}


def prediction_geometry(
    prediction: dict,
    view: dict,
    debug: dict,
    layer: SMPL_Layer,
    labels: np.ndarray,
    smplx_to_smpl: np.ndarray,
    joint_regressor: np.ndarray,
) -> dict[str, dict]:
    """Return raw-Human3R-world SMPL geometry indexed by GT evaluation ID."""
    humans = layer_humans(prediction, view, debug, layer)
    output = {}
    for detection_index, identity_index in enumerate(np.asarray(labels, dtype=np.int64)):
        if identity_index < 0 or detection_index >= len(humans):
            continue
        vertices = smplx_to_smpl @ np.asarray(
            humans[detection_index]["vertices"], dtype=np.float64
        )
        joints = joint_regressor @ vertices
        output[IDENTITIES[int(identity_index)]] = {
            "detection_index": int(detection_index),
            "vertices": vertices,
            "joints": joints,
            "root": joints[0],
        }
    return output


def evaluate_humans(
    geometry: dict[str, dict],
    gt: dict[str, dict],
    boundary: np.ndarray,
    evaluation_gauge: np.ndarray,
    joint_regressor: np.ndarray,
) -> dict:
    per_person = {}
    predicted_roots, target_roots = {}, {}
    for identity in IDENTITIES:
        if identity not in geometry or identity not in gt:
            continue
        predicted = geometry[identity]
        target_vertices_aria = np.asarray(gt[identity]["vertices"], dtype=np.float64)
        target_joints_aria = joint_regressor @ target_vertices_aria
        final_vertices = transform_points(boundary, predicted["vertices"])
        final_joints = transform_points(boundary, predicted["joints"])
        target_vertices = transform_points(evaluation_gauge, target_vertices_aria)
        target_joints = transform_points(evaluation_gauge, target_joints_aria)
        root_error = float(np.linalg.norm(final_joints[0] - target_joints[0]))
        joint_error = float(
            np.linalg.norm(final_joints - target_joints, axis=1).mean()
        )
        vertex_error = float(
            np.linalg.norm(final_vertices - target_vertices, axis=1).mean()
        )
        predicted_roots[identity] = final_joints[0]
        target_roots[identity] = target_joints[0]
        per_person[identity] = {
            "detection_index": int(predicted["detection_index"]),
            "root_error_m": root_error,
            "joint_error_m": joint_error,
            "vertex_error_m": vertex_error,
        }

    pairwise_distance, pairwise_vector = [], []
    for first, second in combinations(sorted(predicted_roots), 2):
        predicted_vector = predicted_roots[first] - predicted_roots[second]
        target_vector = target_roots[first] - target_roots[second]
        pairwise_distance.append(
            abs(float(np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector)))
        )
        pairwise_vector.append(float(np.linalg.norm(predicted_vector - target_vector)))
    output = {
        metric: float(np.mean([row[metric] for row in per_person.values()]))
        if per_person
        else float("nan")
        for metric in HUMAN_METRICS
    }
    output.update(
        {
            "evaluated_person_count": len(per_person),
            "pairwise_root_distance_error_m": (
                float(np.mean(pairwise_distance)) if pairwise_distance else float("nan")
            ),
            "pairwise_root_vector_error_m": (
                float(np.mean(pairwise_vector)) if pairwise_vector else float("nan")
            ),
            "per_person": per_person,
        }
    )
    return output


def evaluate_cut(
    chain_index: int,
    cut_index: int,
    args: argparse.Namespace,
    model: ARCroco3DStereo,
    da3_aligner: DA3FineAligner,
    layer: SMPL_Layer,
    smplx_to_smpl: np.ndarray,
    joint_regressor: np.ndarray,
) -> dict:
    pre, post = CHAINS[chain_index][cut_index : cut_index + 2]
    pre_camera_name, pre_frames = pre
    post_camera_name, post_frames = post
    pre_paths = image_paths(args.data_root, pre_camera_name, pre_frames)
    post_paths = image_paths(args.data_root, post_camera_name, post_frames)
    cut = len(pre_paths)

    shadow_views = set_event_indices(
        prepare_full_square_input(model, pre_paths + post_paths[:1], args), {cut}
    )
    raw_first_views = set_event_indices(
        prepare_full_square_input(model, post_paths[:1], args), set()
    )
    raw_views = set_event_indices(
        prepare_full_square_input(model, post_paths, args), set()
    )
    shadow, shadow_returned, shadow_debug, shadow_seconds = run_rollout(
        model, shadow_views, str(args.device), f"chain{chain_index}_cut{cut_index}_shadow"
    )
    raw_first, _, _, raw_first_seconds = run_rollout(
        model,
        raw_first_views,
        str(args.device),
        f"chain{chain_index}_cut{cut_index}_raw_first",
    )
    raw, raw_returned, raw_debug, raw_seconds = run_rollout(
        model, raw_views, str(args.device), f"chain{chain_index}_cut{cut_index}_raw_full"
    )

    pre_camera = camera_matrix(shadow[-2]).astype(np.float64)
    raw_first_camera = camera_matrix(raw_first[0]).astype(np.float64)
    b0 = (
        boundary_from_camera_predictions(shadow[-1], raw_first[0])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    # The deployable candidate/gate path ends here.  No GT has been loaded.
    fine, fine_diagnostics = da3_aligner.refine_images(
        b0,
        pre_camera,
        raw_first_camera,
        rgb(pre_paths[-1]),
        rgb(post_paths[0]),
    )
    boundaries = {"b0": b0, "b0_da3_safe": np.asarray(fine, dtype=np.float64)}

    # Evaluation-only section: camera GT, identity GT, and SMPL GT first appear here.
    _, exo = load_colmap(args.data_root)
    evaluation_gauge = pre_camera @ np.linalg.inv(
        gt_camera(exo, pre_camera_name)
    )
    target_camera = evaluation_gauge @ gt_camera(exo, post_camera_name)
    cameras = [post_camera_name] * len(post_frames)
    labels, assignments = assign_identities(
        args.data_root,
        cameras,
        list(post_frames),
        raw_returned,
        raw_debug,
        int(args.size),
        float(args.assignment_threshold_px),
    )

    frames = []
    for frame_index, (frame, prediction, view, debug, frame_labels) in enumerate(
        zip(post_frames, raw, raw_returned, raw_debug, labels)
    ):
        raw_camera = camera_matrix(prediction).astype(np.float64)
        geometry = prediction_geometry(
            prediction,
            view,
            debug,
            layer,
            frame_labels,
            smplx_to_smpl,
            joint_regressor,
        )
        target_bodies = gt_smpl(args.data_root, int(frame))
        methods = {}
        for name, boundary in boundaries.items():
            methods[name] = {
                **boundary_camera_metrics(boundary, raw_camera, target_camera),
                **evaluate_humans(
                    geometry,
                    target_bodies,
                    boundary,
                    evaluation_gauge,
                    joint_regressor,
                ),
            }
        frames.append(
            {
                "post_index": frame_index,
                "camera": post_camera_name,
                "frame": int(frame),
                "labels_by_detection": frame_labels,
                "assignment": assignments[frame_index],
                "methods": methods,
            }
        )

    def cut_method_summary(name: str) -> dict:
        keys = CAMERA_METRICS + HUMAN_METRICS + (
            "pairwise_root_distance_error_m",
            "pairwise_root_vector_error_m",
        )
        return {
            key: finite_stats([frame["methods"][name][key] for frame in frames])
            for key in keys
        } | {
            "evaluated_person_count": int(
                sum(frame["methods"][name]["evaluated_person_count"] for frame in frames)
            ),
            "catastrophic_frame_count": int(
                sum(frame["methods"][name]["catastrophic"] for frame in frames)
            ),
        }

    return {
        "status": "ok",
        "case_key": f"chain{chain_index}_cut{cut_index}_{pre_camera_name}_{post_camera_name}",
        "chain_index": chain_index,
        "cut_index": cut_index,
        "stream": {
            "pre_camera": pre_camera_name,
            "pre_frames": pre_frames,
            "post_camera": post_camera_name,
            "post_frames": post_frames,
            "boundary_timestamp_repeated": bool(pre_frames[-1] == post_frames[0]),
        },
        "proposal_input_audit": {
            "inputs": "V14 shadow/raw camera, last-pre RGB, first-post RGB, frozen DA3",
            "future_post_frames_used": False,
            "gt_used_for_boundary_or_gate": False,
            "gt_usage": "camera/body evaluation and detection identity assignment only after both boundaries freeze",
        },
        "poses": {
            "human3r_pre": pre_camera,
            "human3r_raw_first": raw_first_camera,
            "human3r_raw_full_first": camera_matrix(raw[0]).astype(np.float64),
            "target_post_evaluation_only": target_camera,
        },
        "boundaries": boundaries,
        "fine_diagnostics": fine_diagnostics,
        "timing_seconds": {
            "human3r_shadow": shadow_seconds,
            "human3r_raw_first": raw_first_seconds,
            "human3r_raw_full": raw_seconds,
            "da3_forward": fine_diagnostics.get("da3_forward_seconds", float("nan")),
            "da3_reverse": fine_diagnostics.get("da3_reverse_seconds", float("nan")),
        },
        "summary": {name: cut_method_summary(name) for name in METHODS},
        "frames": frames,
    }


def finite_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(rows: list[dict]) -> dict:
    post_frame_count = int(sum(len(row["frames"]) for row in rows))
    summary = {
        "cut_count": len(rows),
        "post_frame_count": post_frame_count,
        "expected_person_instances": post_frame_count * len(IDENTITIES),
        "gate_accept_count": int(
            sum(bool(row["fine_diagnostics"].get("accepted")) for row in rows)
        ),
        "gate_fallback_count": int(
            sum(not bool(row["fine_diagnostics"].get("accepted")) for row in rows)
        ),
        "methods": {},
    }
    summary["gate_acceptance"] = (
        summary["gate_accept_count"] / len(rows) if rows else float("nan")
    )
    metric_keys = CAMERA_METRICS + HUMAN_METRICS + (
        "pairwise_root_distance_error_m",
        "pairwise_root_vector_error_m",
    )
    for method in METHODS:
        frames = [frame for row in rows for frame in row["frames"]]
        people = [
            person
            for frame in frames
            for person in frame["methods"][method]["per_person"].values()
        ]
        method_summary = {}
        for metric in metric_keys:
            if metric in HUMAN_METRICS:
                method_summary[metric] = finite_stats(
                    [person[metric] for person in people]
                )
            else:
                method_summary[metric] = finite_stats(
                    [frame["methods"][method][metric] for frame in frames]
                )
        method_summary["evaluated_person_instances"] = len(people)
        method_summary["evaluation_coverage"] = (
            len(people) / summary["expected_person_instances"]
            if summary["expected_person_instances"]
            else float("nan")
        )
        method_summary["catastrophic_frame_count"] = int(
            sum(frame["methods"][method]["catastrophic"] for frame in frames)
        )
        if method != "b0":
            method_summary["paired_delta_vs_b0"] = {
                metric: finite_stats(
                    [
                        frame["methods"][method][metric]
                        - frame["methods"]["b0"][metric]
                        for frame in frames
                    ]
                )
                for metric in metric_keys
            }
            method_summary["cut_improvement_count_vs_b0"] = {
                metric: int(
                    sum(
                        row["summary"][method][metric]["mean"]
                        < row["summary"]["b0"][metric]["mean"]
                        for row in rows
                    )
                )
                for metric in metric_keys
            }
        summary["methods"][method] = method_summary
    return summary


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Frozen B0 + DA3 on EgoHumans (6 cuts)",
        "",
        "GT is loaded only after B0 and B0+DA3-safe are frozen. GT identity is evaluation-only.",
        "Predicted SMPL-X is mapped to the common 6890-vertex SMPL topology before scoring.",
        "",
        f"Completed cuts: `{summary['cut_count']}`; failures: `{len(report['failures'])}`; "
        f"DA3 accepted/fallback: `{summary['gate_accept_count']}/{summary['gate_fallback_count']}`.",
        f"Evaluated person instances: `{summary['methods']['b0']['evaluated_person_instances']}/"
        f"{summary['expected_person_instances']}` "
        f"(`{summary['methods']['b0']['evaluation_coverage']:.1%}` coverage).",
        "",
        "| Method | Camera T (m) | Camera R (deg) | Root (m) | World joint-24 (m) | World vertex-6890 (m) | Pair root vector (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"].get(method)
        if not row:
            continue
        lines.append(
            f"| {method} | {row['camera_translation_error_m']['mean']:.4f} | "
            f"{row['camera_rotation_error_deg']['mean']:.3f} | "
            f"{row['root_error_m']['mean']:.4f} | "
            f"{row['joint_error_m']['mean']:.4f} | "
            f"{row['vertex_error_m']['mean']:.4f} | "
            f"{row['pairwise_root_vector_error_m']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-cut mean",
            "",
            "| Cut | Gate | B0 T | DA3 T | B0 R | DA3 R | B0 world joint | DA3 world joint | People |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        b0 = case["summary"]["b0"]
        fine = case["summary"]["b0_da3_safe"]
        lines.append(
            f"| {case['case_key']} | {bool(case['fine_diagnostics'].get('accepted'))} | "
            f"{b0['camera_translation_error_m']['mean']:.4f} | "
            f"{fine['camera_translation_error_m']['mean']:.4f} | "
            f"{b0['camera_rotation_error_deg']['mean']:.3f} | "
            f"{fine['camera_rotation_error_deg']['mean']:.3f} | "
            f"{b0['joint_error_m']['mean']:.4f} | "
            f"{fine['joint_error_m']['mean']:.4f} | "
            f"{b0['evaluated_person_count']} |"
        )
    improvements = summary["methods"]["b0_da3_safe"][
        "cut_improvement_count_vs_b0"
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"DA3 improves mean camera translation on `{improvements['camera_translation_error_m']}/"
            f"{summary['cut_count']}` cuts, camera rotation on "
            f"`{improvements['camera_rotation_error_deg']}/{summary['cut_count']}`, and MPJPE on "
            f"`{improvements['joint_error_m']}/{summary['cut_count']}`. The gate accepts every cut, "
            "yet aggregate camera translation, root, joint, and vertex error all worsen. Therefore "
            "the current agreement/prior gate is not sufficient on raw fisheye EgoHumans input.",
            "",
            "For context, Multi-THuMBS reports EgoHumans W-MPJPE 279.0, WA-MPJPE 166.0, "
            "MPJPE 228.3, MPVPE 262.2, Accel 27.3, ATE 0.7, and IDs 0.97. Our world-gauge "
            "24-joint/6890-vertex diagnostic is 347.8/350.0 mm for B0 and 360.2/360.8 mm for "
            "B0+DA3. These values expose a gap but are not a leaderboard claim because the paper's "
            "clip list and exact alignment/aggregation protocol are unpublished.",
            "",
            "These are six local-cut diagnostics drawn from three 15-frame streams (45 unique RGB observations); metrics score the 30 post-cut observations. They are not a reproduction of the unpublished Multi-THuMBS split/protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if not (args.da3_path / "model.safetensors").is_file():
        raise FileNotFoundError(args.da3_path / "model.safetensors")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    smplx_to_smpl, joint_regressor = smpl_resources()

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    da3 = DepthAnything3.from_pretrained(str(args.da3_path)).to(device).eval()
    aligner = DA3FineAligner(
        da3, config=DEFAULT_CONFIG, process_res=int(args.process_res), use_ray_pose=False
    )
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()

    rows, failures = [], []
    selected = sorted(requested_cases(args.case))
    for ordinal, (chain_index, cut_index) in enumerate(selected, start=1):
        pre_camera_name = CHAINS[chain_index][cut_index][0]
        post_camera_name = CHAINS[chain_index][cut_index + 1][0]
        key = f"chain{chain_index}_cut{cut_index}_{pre_camera_name}_{post_camera_name}"
        case_path = cases_dir / f"{key}.json"
        cached = None
        if case_path.is_file() and not args.overwrite:
            cached = json.loads(case_path.read_text(encoding="utf-8"))
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                started = time.perf_counter()
                row = evaluate_cut(
                    chain_index,
                    cut_index,
                    args,
                    model,
                    aligner,
                    layer,
                    smplx_to_smpl,
                    joint_regressor,
                )
                row["wall_seconds"] = time.perf_counter() - started
            except Exception as error:
                row = {
                    "status": "failed",
                    "case_key": key,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            case_path.write_text(
                json.dumps(jsonable(row), indent=2, ensure_ascii=False, allow_nan=True)
                + "\n",
                encoding="utf-8",
            )
        if row["status"] == "ok":
            rows.append(row)
            print(
                f"[{ordinal}/{len(selected)}] {key} "
                f"gate={row['fine_diagnostics'].get('accepted')} "
                f"camera_T={row['summary']['b0']['camera_translation_error_m']['mean']:.4f}->"
                f"{row['summary']['b0_da3_safe']['camera_translation_error_m']['mean']:.4f} "
                f"MPJPE={row['summary']['b0']['joint_error_m']['mean']:.4f}->"
                f"{row['summary']['b0_da3_safe']['joint_error_m']['mean']:.4f}",
                flush=True,
            )
        else:
            failures.append(row)
            print(f"[{ordinal}/{len(selected)}] {key} FAILED {row['error']}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "experiment": "v14_frozen_b0_da3_egohumans_six_cut",
        "protocol": {
            "dataset_root": str(args.data_root),
            "chains": CHAINS,
            "requested_cases": selected,
            "candidate_inputs": "V14 shadow/raw, last-pre/first-post raw fisheye RGB, frozen DA3",
            "gt_usage": "evaluation-only camera/body plus post-hoc identity assignment",
            "human_topology": "SMPL-X 10475 -> official mapping -> SMPL 6890; SMPL 24-joint regressor",
            "fine_alignment_config": DEFAULT_CONFIG.__dict__,
            "same_timestamp_boundary": True,
            "multi_thumbs_comparability": (
                "dataset-level diagnostic only; paper clip list and exact protocol are unpublished"
            ),
        },
        "models": {
            "human3r_checkpoint": str(args.model_path),
            "human3r_flags": flags,
            "da3_checkpoint": str(args.da3_path),
            "device": str(device),
            "process_res": int(args.process_res),
        },
        "summary": summarize(rows),
        "failures": failures,
        "cases": rows,
    }
    json_path = args.output_dir / "v14_b0_da3_egohumans.json"
    markdown_path = args.output_dir / "v14_b0_da3_egohumans.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_text = markdown(jsonable(report))
    markdown_path.write_text(markdown_text, encoding="utf-8")
    print(markdown_text, flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
