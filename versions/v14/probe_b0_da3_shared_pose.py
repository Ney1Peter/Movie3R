#!/usr/bin/env python3
"""Probe frozen DA3-Base shared-space pose as a residual around frozen B0.

Deployment inputs are the last pre-cut RGB, first post-cut RGB, predicted Human3R
boxes, frozen B0, and the two raw Human3R camera poses. GT is used only to rebuild
strict evaluation identities and score the resulting shared Boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset" / "Depth-Anything-3"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(DA3_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)


from depth_anything_3.api import DepthAnything3  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.b0_da3_fine_alignment import (  # noqa: E402
    DEFAULT_CONFIG,
    refine_b0_with_da3,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_residual_observability import (  # noqa: E402
    evaluate_boundary,
    serializable,
    transform,
    vector_stats,
)
from versions.v14.probe_b0_sift_epipolar import (  # noqa: E402
    FrameReader,
    background_mask,
    boundary_from_camera_center,
    slerp_direction,
)
from versions.v14.run_v14_2_multihuman_sequence import solution  # noqa: E402,F401


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/da3_shared_pose"
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_MODEL = DA3_ROOT / "checkpoints" / "DAE-base"
ROTATION_CAPS_DEG = (1, 2, 3, 5, 10)
DIRECTION_CAPS_DEG = (2, 5, 10, 20)
COMBINED_CAPS_DEG = ((1, 2), (2, 2), (2, 5), (3, 5))
SAFE_ROTATION_SPREAD_DEG = DEFAULT_CONFIG.rotation_spread_limit_deg
SAFE_DIRECTION_SPREAD_DEG = DEFAULT_CONFIG.direction_spread_limit_deg
SAFE_RIGHT_ROTATION_DEG = DEFAULT_CONFIG.right_rotation_limit_deg
SAFE_DIRECTION_VS_B0_DEG = DEFAULT_CONFIG.direction_vs_b0_limit_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", nargs="+", choices=tuple(SEQUENCE_INPUTS), default=("three",)
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--human_margin", type=float, default=-0.25)
    parser.add_argument(
        "--image_modes",
        nargs="+",
        choices=("full", "background"),
        default=("full", "background"),
    )
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--skip_identity_rebuild", action="store_true")
    return parser.parse_args()


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected 3x4 or 4x4 pose, got {matrix.shape}")
    output = np.eye(4, dtype=np.float64)
    output[:3] = matrix
    return output


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])))


def direction_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return float(math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0)))))


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    value = np.sum([np.asarray(row, dtype=np.float64) for row in rotations], axis=0)
    left, _, right = np.linalg.svd(value)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(left @ right)
    return left @ correction @ right


def mean_direction(directions: list[np.ndarray]) -> np.ndarray | None:
    rows = []
    for value in directions:
        value = np.asarray(value, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm > 1e-12:
            rows.append(value / norm)
    if not rows:
        return None
    output = np.sum(rows, axis=0)
    if float(np.linalg.norm(output)) < 1e-12:
        return None
    return output / np.linalg.norm(output)


def whiten_humans(
    image_rgb: np.ndarray, humans: dict[str, dict], margin_fraction: float
) -> np.ndarray:
    output = image_rgb.copy()
    mask = background_mask(humans, image_rgb.shape[0], margin_fraction)
    output[mask == 0] = 255
    return output


def run_da3_pair(
    model: DepthAnything3,
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    prediction = model.inference(
        [first_rgb, second_rgb],
        process_res=int(args.process_res),
        use_ray_pose=bool(args.use_ray_pose),
        ref_view_strategy="first",
    )
    elapsed = time.perf_counter() - started
    if prediction.extrinsics is None:
        return {"status": "no_extrinsics", "elapsed_seconds": elapsed}
    extrinsics = np.stack(
        [homogeneous(row) for row in np.asarray(prediction.extrinsics)]
    )
    camera_to_world = np.linalg.inv(extrinsics)
    output = {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "camera_to_world": camera_to_world,
        "baseline_units": float(
            np.linalg.norm(camera_to_world[1, :3, 3] - camera_to_world[0, :3, 3])
        ),
    }
    if prediction.intrinsics is not None:
        output["intrinsics"] = np.asarray(prediction.intrinsics)
    if prediction.conf is not None:
        confidence = np.asarray(prediction.conf, dtype=np.float64)
        output["confidence_mean"] = float(np.mean(confidence))
        output["confidence_p10"] = float(np.percentile(confidence, 10))
    return output


def proposal_from_prediction(
    cache: dict, prediction: dict, reverse: bool
) -> dict | None:
    if prediction["status"] != "ok":
        return None
    camera_to_world = np.asarray(prediction["camera_to_world"], dtype=np.float64)
    pre_index, post_index = ((1, 0) if reverse else (0, 1))
    da3_pre = camera_to_world[pre_index]
    da3_post = camera_to_world[post_index]
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    pre_world_from_da3 = pre_pose @ np.linalg.inv(da3_pre)
    desired_post_pose = pre_world_from_da3 @ da3_post
    boundary = desired_post_pose @ np.linalg.inv(post_pose)
    baseline = desired_post_pose[:3, 3] - pre_pose[:3, 3]
    if float(np.linalg.norm(baseline)) < 1e-12:
        return None
    return {
        "boundary_rotation": boundary[:3, :3],
        "baseline_direction_world": baseline / np.linalg.norm(baseline),
        "da3_baseline_units": float(np.linalg.norm(baseline)),
    }


def consensus_proposal(proposals: list[dict]) -> dict | None:
    if not proposals:
        return None
    direction = mean_direction(
        [proposal["baseline_direction_world"] for proposal in proposals]
    )
    if direction is None:
        return None
    return {
        "boundary_rotation": mean_rotation(
            [proposal["boundary_rotation"] for proposal in proposals]
        ),
        "baseline_direction_world": direction,
        "da3_baseline_units": float(
            np.median([proposal["da3_baseline_units"] for proposal in proposals])
        ),
    }


def safe_gate_decision(diagnostics: dict) -> bool:
    required = (
        "forward_reverse_rotation_spread_deg",
        "forward_reverse_direction_spread_deg",
        "right_rotation_deg",
        "direction_vs_b0_deg",
    )
    if any(key not in diagnostics for key in required):
        return False
    values = np.asarray([diagnostics[key] for key in required], dtype=np.float64)
    if not np.isfinite(values).all():
        return False
    return bool(
        diagnostics["forward_reverse_rotation_spread_deg"]
        <= SAFE_ROTATION_SPREAD_DEG
        and diagnostics["forward_reverse_direction_spread_deg"]
        <= SAFE_DIRECTION_SPREAD_DEG
        and diagnostics["right_rotation_deg"] <= SAFE_RIGHT_ROTATION_DEG
        and diagnostics["direction_vs_b0_deg"] <= SAFE_DIRECTION_VS_B0_DEG
    )


def add_proposal_methods(
    methods: dict,
    name: str,
    proposal: dict,
    cache: dict,
    b0: np.ndarray,
    identities: tuple[str, ...],
) -> dict:
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    coarse_center = geometry.transform_points(b0, post_pose[:3, 3][None])[0]
    coarse_baseline = coarse_center - pre_pose[:3, 3]
    coarse_length = float(np.linalg.norm(coarse_baseline))
    right_rotation = b0[:3, :3].T @ proposal["boundary_rotation"]
    right_rotvec = cv2.Rodrigues(right_rotation)[0].reshape(3)

    methods[f"{name}_rotation"] = evaluate_boundary(
        cache,
        boundary_from_camera_center(
            cache, proposal["boundary_rotation"], coarse_center
        ),
        identities,
    )
    bounded_rotations = {}
    for cap_deg in ROTATION_CAPS_DEG:
        maximum = math.radians(float(cap_deg))
        norm = float(np.linalg.norm(right_rotvec))
        bounded_vector = right_rotvec * min(1.0, maximum / max(norm, 1e-12))
        bounded = cv2.Rodrigues(bounded_vector)[0]
        bounded_rotations[cap_deg] = b0[:3, :3] @ bounded
        methods[f"{name}_rotation_b{cap_deg}"] = evaluate_boundary(
            cache,
            boundary_from_camera_center(
                cache, bounded_rotations[cap_deg], coarse_center
            ),
            identities,
        )

    bounded_directions = {}
    for cap_deg in DIRECTION_CAPS_DEG:
        direction = slerp_direction(
            coarse_baseline, proposal["baseline_direction_world"], cap_deg
        )
        bounded_directions[cap_deg] = direction
        center = pre_pose[:3, 3] + coarse_length * direction
        methods[f"{name}_direction_b{cap_deg}"] = evaluate_boundary(
            cache,
            boundary_from_camera_center(cache, b0[:3, :3], center),
            identities,
        )

    for rotation_cap, direction_cap in COMBINED_CAPS_DEG:
        center = (
            pre_pose[:3, 3]
            + coarse_length * bounded_directions[direction_cap]
        )
        methods[
            f"{name}_combined_r{rotation_cap}_d{direction_cap}"
        ] = evaluate_boundary(
            cache,
            boundary_from_camera_center(
                cache, bounded_rotations[rotation_cap], center
            ),
            identities,
        )
    return {
        "right_rotation_deg": float(np.degrees(np.linalg.norm(right_rotvec))),
        "direction_vs_b0_deg": direction_angle_deg(
            coarse_baseline, proposal["baseline_direction_world"]
        ),
        "da3_baseline_units": float(proposal["da3_baseline_units"]),
    }


def load_case(
    sequence: str,
    report_case: dict,
    reader: FrameReader,
    model: DepthAnything3,
    args: argparse.Namespace,
) -> dict:
    inputs = SEQUENCE_INPUTS[sequence]
    case = report_case["case"]
    cache = torch.load(
        inputs["cache"] / f"{case['key']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    if not args.skip_identity_rebuild:
        cache = geometry.reassign_cache_gt_identities(
            SimpleNamespace(data_root=args.data_root, size=512, sequence=sequence),
            cache,
        )
    first = reader.read(
        int(case["source_camera"]), int(case["pre_frames"][-1])
    )[..., ::-1].copy()
    second = reader.read(
        int(case["target_camera"]), int(case["post_frame"])
    )[..., ::-1].copy()
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    identities = tuple(
        identity for identity in geometry.IDENTITIES if identity in cache["humans"][-1]
    )
    methods = {"b0": evaluate_boundary(cache, b0, identities)}
    predictions, proposals, proposal_diagnostics = {}, {}, {}
    for image_mode in args.image_modes:
        if image_mode == "background":
            first_input = whiten_humans(
                first, cache["humans"][-2], float(args.human_margin)
            )
            second_input = whiten_humans(
                second, cache["humans"][-1], float(args.human_margin)
            )
        else:
            first_input, second_input = first, second
        directions = (("fwd", False),)
        if not args.forward_only:
            directions += (("rev", True),)
        current = []
        for direction_name, reverse in directions:
            prediction = run_da3_pair(
                model,
                second_input if reverse else first_input,
                first_input if reverse else second_input,
                args,
            )
            key = f"{image_mode}_{direction_name}"
            predictions[key] = prediction
            proposal = proposal_from_prediction(cache, prediction, reverse)
            if proposal is None:
                continue
            proposals[key] = proposal
            current.append(proposal)
            proposal_diagnostics[key] = add_proposal_methods(
                methods, key, proposal, cache, b0, identities
            )
        consensus = consensus_proposal(current)
        if consensus is not None and len(current) > 1:
            key = f"{image_mode}_fb"
            proposals[key] = consensus
            proposal_diagnostics[key] = add_proposal_methods(
                methods, key, consensus, cache, b0, identities
            )
            proposal_diagnostics[key]["forward_reverse_rotation_spread_deg"] = (
                rotation_angle_deg(
                    current[0]["boundary_rotation"],
                    current[1]["boundary_rotation"],
                )
            )
            proposal_diagnostics[key]["forward_reverse_direction_spread_deg"] = (
                direction_angle_deg(
                    current[0]["baseline_direction_world"],
                    current[1]["baseline_direction_world"],
                )
            )
    forward = predictions.get("full_fwd", {})
    reverse = predictions.get("full_rev", {})
    safe_boundary, safe_diagnostics = refine_b0_with_da3(
        b0,
        np.asarray(cache["poses"][-2], dtype=np.float64),
        np.asarray(cache["poses"][-1], dtype=np.float64),
        (
            np.asarray(forward["camera_to_world"], dtype=np.float64)
            if forward.get("status") == "ok"
            else None
        ),
        (
            np.asarray(reverse["camera_to_world"], dtype=np.float64)
            if reverse.get("status") == "ok"
            else None
        ),
    )
    methods["da3_safe"] = evaluate_boundary(
        cache, safe_boundary, identities
    )
    proposal_diagnostics["da3_safe"] = safe_diagnostics
    all_consensus = consensus_proposal(list(proposals.values()))
    if all_consensus is not None and len(proposals) > 1:
        proposal_diagnostics["all"] = add_proposal_methods(
            methods, "all", all_consensus, cache, b0, identities
        )
        proposal_diagnostics["all"]["rotation_spread_deg"] = float(
            max(
                rotation_angle_deg(
                    all_consensus["boundary_rotation"],
                    proposal["boundary_rotation"],
                )
                for proposal in proposals.values()
            )
        )
        proposal_diagnostics["all"]["direction_spread_deg"] = float(
            max(
                direction_angle_deg(
                    all_consensus["baseline_direction_world"],
                    proposal["baseline_direction_world"],
                )
                for proposal in proposals.values()
            )
        )
    return {
        "sequence": sequence,
        "case": case,
        "camera_span_deg": float(report_case["camera_span_deg"]),
        "predictions": predictions,
        "proposal_diagnostics": proposal_diagnostics,
        "methods": methods,
    }


def summarize(rows: list[dict]) -> dict:
    method_names = sorted(set().union(*(set(row["methods"]) for row in rows)))
    output = {"case_count": len(rows), "methods": {}}
    for method in method_names:
        method_rows = [row for row in rows if method in row["methods"]]
        values = {
            metric: vector_stats(
                [row["methods"][method][metric] for row in method_rows]
            )
            for metric in (
                "camera_translation_error_m",
                "camera_rotation_error_deg",
                "camera_composite",
                "human_root_error_m",
            )
        }
        values["valid_cases"] = len(method_rows)
        values["paired"] = {
            metric: {
                "mean_delta": float(
                    np.mean(
                        [
                            row["methods"][method][metric]
                            - row["methods"]["b0"][metric]
                            for row in method_rows
                        ]
                    )
                ),
                "improvement_rate": float(
                    np.mean(
                        [
                            row["methods"][method][metric]
                            < row["methods"]["b0"][metric]
                            for row in method_rows
                        ]
                    )
                ),
            }
            for metric in (
                "camera_rotation_error_deg",
                "camera_composite",
                "human_root_error_m",
            )
        }
        output["methods"][method] = values
    output["runtime_seconds"] = vector_stats(
        [
            prediction["elapsed_seconds"]
            for row in rows
            for prediction in row["predictions"].values()
        ]
    )
    return output


def write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    methods = summary["methods"]
    ranked = sorted(
        methods,
        key=lambda name: methods[name]["camera_composite"]["mean"],
    )
    shown = ["b0"] + [name for name in ranked if name != "b0"][:35]
    lines = [
        "# V14 B0 + DA3 Shared-Pose Probe",
        "",
        f"Sequences: `{', '.join(report['sequences'])}`; cases: "
        f"`{summary['case_count']}`.",
        "",
        "DA3-Base receives only the last pre-cut and first post-cut RGB. Predicted "
        "Human3R boxes optionally whiten the central human region. DA3 translation "
        "scale is discarded: direction candidates retain B0's baseline length. GT "
        "is evaluation only.",
        "",
        "| Method | N | Camera T | Camera R | Composite | Human root | Composite improve |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in shown:
        values = methods[method]
        paired = values["paired"]["camera_composite"]
        lines.append(
            f"| {method} | {values['valid_cases']} | "
            f"{values['camera_translation_error_m']['mean']:.4f} | "
            f"{values['camera_rotation_error_deg']['mean']:.3f} | "
            f"{values['camera_composite']['mean']:.4f} | "
            f"{values['human_root_error_m']['mean']:.4f} | "
            f"{100 * paired['improvement_rate']:.1f}% "
            f"(`{paired['mean_delta']:+.4f}`) |"
        )
    lines.extend(
        [
            "",
            f"DA3 pair runtime mean/P95: "
            f"`{summary['runtime_seconds']['mean']:.3f}/"
            f"{summary['runtime_seconds']['p95']:.3f} s`.",
            "",
            "The JSON artifact contains every method, paired delta, proposal spread, "
            "and per-case prediction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("DA3 shared-pose probe requires CUDA")
    if not (args.model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Incomplete DA3 checkpoint: {args.model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for sequence in args.sequences:
        reader = FrameReader(args, sequence)
        try:
            source = json.loads(SEQUENCE_INPUTS[sequence]["report"].read_text())
            report_cases = source["cases"][: args.max_cases or None]
            for index, report_case in enumerate(report_cases, start=1):
                row = load_case(sequence, report_case, reader, model, args)
                rows.append(row)
                print(
                    f"[{sequence} {index:03d}/{len(report_cases):03d}] "
                    f"{report_case['case']['key']} methods={len(row['methods'])}",
                    flush=True,
                )
        finally:
            reader.close()
    report = {
        "experiment": "v14_b0_da3_shared_pose",
        "sequences": list(args.sequences),
        "protocol": {
            "rgb": "last pre-cut plus first post-cut frame",
            "model": "frozen DA3-Base any-view pose model",
            "image_modes": list(args.image_modes),
            "bidirectional": not args.forward_only,
            "translation": "direction only; B0 metric baseline length retained",
            "gt_usage": "strict identity rebuild and evaluation only",
        },
        "parameters": {
            "model_path": str(args.model_path),
            "device": str(args.device),
            "resolution": int(args.resolution),
            "process_res": int(args.process_res),
            "human_margin": float(args.human_margin),
            "use_ray_pose": bool(args.use_ray_pose),
            "rotation_caps_deg": ROTATION_CAPS_DEG,
            "direction_caps_deg": DIRECTION_CAPS_DEG,
            "combined_caps_deg": COMBINED_CAPS_DEG,
            "safe_gate": {
                "rotation_spread_deg": SAFE_ROTATION_SPREAD_DEG,
                "direction_spread_deg": SAFE_DIRECTION_SPREAD_DEG,
                "right_rotation_deg": SAFE_RIGHT_ROTATION_DEG,
                "direction_vs_b0_deg": SAFE_DIRECTION_VS_B0_DEG,
                "accepted_candidate": "full_fb_combined_r3_d5",
                "fallback": "exact frozen B0",
            },
        },
        "summary": summarize(rows),
        "cases": rows,
    }
    json_path = args.output_dir / "v14_b0_da3_shared_pose.json"
    md_path = args.output_dir / "v14_b0_da3_shared_pose.md"
    json_path.write_text(
        json.dumps(serializable(report), indent=2), encoding="utf-8"
    )
    write_markdown(md_path, report)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
