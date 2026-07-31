#!/usr/bin/env python3
"""Strict root-ray person mesh/DA3 depth probe with a frozen B0 camera.

This is a non-destructive v2 of ``probe_b0_da3_person_mesh_depth.py``.  For each
same-pixel visible SMPL-X surface and DA3 depth sample, both observations are
unprojected to post-camera 3D with their own intrinsics.  The candidate scalar
is ``dot(x_da3 - x_mesh, root_ray_camera)``.  Only a rigid translation along
the current root ray is permitted.  GT is accessed only after candidates and
gates have frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_da3_person_mesh_depth as v1  # noqa: E402


DEFAULT_DATA = v1.DEFAULT_DATA
DEFAULT_MODEL = v1.DEFAULT_MODEL
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_da3_person_mesh_depth_v2"
)
METHODS = ("b0", "mesh_depth_v2_translation_cap030", "oracle_gt_ray_translation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--cap_m", type=float, default=0.30)
    parser.add_argument("--min_pixels", type=int, default=96)
    parser.add_argument("--max_residual_mad_m", type=float, default=0.25)
    parser.add_argument("--min_median_to_mad", type=float, default=2.0)
    parser.add_argument("--erode_iterations", type=int, default=1)
    return parser.parse_args()


def unproject_selected(
    z_depth: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Unproject selected z-depth pixels to camera 3D without ray normalization."""
    yy, xx = np.nonzero(valid)
    pixels = np.stack(
        [xx.astype(np.float64), yy.astype(np.float64), np.ones(len(xx))], axis=1
    )
    rays = (np.linalg.inv(np.asarray(intrinsic, dtype=np.float64)) @ pixels.T).T
    return rays * np.asarray(z_depth[valid], dtype=np.float64)[:, None]


def observed_root_ray_residual(
    mesh_z: np.ndarray,
    all_mesh_z: np.ndarray,
    mesh_intrinsic: np.ndarray,
    da3_depth: np.ndarray,
    da3_intrinsic: np.ndarray,
    da3_confidence: np.ndarray,
    depth_scale: float,
    root_ray_camera: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Return strict same-pixel ``dot(x_da3-x_mesh, root_ray)`` samples."""
    finite = np.isfinite(mesh_z)
    visible = finite & (mesh_z <= all_mesh_z + 1e-4)
    if args.erode_iterations:
        visible = cv2.erode(
            visible.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=int(args.erode_iterations),
        ).astype(bool)
    valid = (
        visible
        & np.isfinite(da3_depth)
        & (da3_depth > 0.02)
        & np.isfinite(da3_confidence)
    )
    valid_before_confidence = int(valid.sum())
    common = {
        "silhouette_pixels": int(finite.sum()),
        "visible_pixels": int(visible.sum()),
        "preconfidence_valid_pixels": valid_before_confidence,
    }
    if valid_before_confidence < int(args.min_pixels):
        return {
            "accepted": False,
            "reason": "too_few_same_surface_pixels",
            "rejection_reasons": ["too_few_same_surface_pixels"],
            "valid_pixels": valid_before_confidence,
            **common,
        }

    confidence_threshold = float(np.percentile(da3_confidence[valid], 30.0))
    valid &= da3_confidence >= confidence_threshold
    valid_pixels = int(valid.sum())
    if valid_pixels < int(args.min_pixels):
        return {
            "accepted": False,
            "reason": "too_few_after_confidence",
            "rejection_reasons": ["too_few_after_confidence"],
            "valid_pixels": valid_pixels,
            "confidence_threshold": confidence_threshold,
            **common,
        }

    mesh_points = unproject_selected(mesh_z, mesh_intrinsic, valid)
    da3_points = unproject_selected(da3_depth, da3_intrinsic, valid) * float(
        depth_scale
    )
    ray = np.asarray(root_ray_camera, dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-8)
    difference = da3_points - mesh_points
    residual = difference @ ray
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    tangential = np.linalg.norm(difference - residual[:, None] * ray[None], axis=1)

    rejection_reasons = []
    if not (0.05 < float(depth_scale) < 20.0):
        rejection_reasons.append("depth_scale_out_of_range")
    if mad > float(args.max_residual_mad_m):
        rejection_reasons.append("root_ray_residual_mad_too_large")
    if not (
        abs(median) > float(args.min_median_to_mad) * max(mad, 0.0)
    ):
        rejection_reasons.append("root_ray_residual_sign_not_reliable")
    accepted = not rejection_reasons
    mesh_projection = mesh_points @ ray
    da3_projection = da3_points @ ray
    mesh_ranges = np.linalg.norm(mesh_points, axis=1)
    da3_ranges = np.linalg.norm(da3_points, axis=1)
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else rejection_reasons[0],
        "rejection_reasons": rejection_reasons,
        **common,
        "valid_pixels": valid_pixels,
        "visible_fraction_of_silhouette": (
            float(common["visible_pixels"] / common["silhouette_pixels"])
            if common["silhouette_pixels"]
            else 0.0
        ),
        "retained_fraction_of_visible": (
            float(valid_pixels / common["visible_pixels"])
            if common["visible_pixels"]
            else 0.0
        ),
        "confidence_threshold": confidence_threshold,
        "confidence_mean": float(np.mean(da3_confidence[valid])),
        "predicted_mesh_surface_range_median_m": float(np.median(mesh_ranges)),
        "scaled_da3_surface_range_median_m": float(np.median(da3_ranges)),
        "predicted_mesh_root_ray_projection_median_m": float(
            np.median(mesh_projection)
        ),
        "scaled_da3_root_ray_projection_median_m": float(
            np.median(da3_projection)
        ),
        "root_ray_residual_median_m": median,
        "root_ray_residual_mad_m": mad,
        "median_to_mad_ratio": abs(median) / max(mad, 1e-8),
        "tangential_difference_median_m": float(np.median(tangential)),
    }


def build_candidates(
    report_case: dict,
    cache: dict,
    da3: dict,
    faces: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict]:
    """GT-free v2 proposal; no ``cache['gt']`` access is permitted here."""
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    final_camera = b0 @ post_pose
    camera_snapshot = final_camera.copy()
    height, width = da3["depth"][1].shape
    human3r_intrinsic = v1.recovered_human3r_intrinsics()
    mapping = v1.mapping_diagnostic(cache, human3r_intrinsic, (height, width))
    if mapping["status"] != "ok":
        raise RuntimeError(f"Unreliable Human3R→DA3 mapping: {mapping}")
    raster_intrinsic = v1.transform_intrinsics_pixel_centers(
        human3r_intrinsic, (v1.HUMAN3R_SIZE, v1.HUMAN3R_SIZE), (height, width)
    )

    da3_c2w = np.linalg.inv(da3["extrinsics"])
    da3_baseline = float(
        np.linalg.norm(da3_c2w[1, :3, 3] - da3_c2w[0, :3, 3])
    )
    frozen_baseline = float(
        np.linalg.norm(final_camera[:3, 3] - pre_pose[:3, 3])
    )
    depth_scale = frozen_baseline / max(da3_baseline, 1e-8)

    pairs = v1.auto_identity_pairs(report_case, cache)
    post_identities = tuple(dict.fromkeys(post for _, post in pairs))
    zbuffers = {}
    local_roots = {}
    for identity in post_identities:
        human = cache["humans"][-1][identity]
        vertices_camera = v1.camera_vertices(human["vertices"], post_pose)
        local_roots[identity] = v1.camera_vertices(
            np.asarray(human["root"])[None], post_pose
        )[0]
        zbuffers[identity] = v1.rasterize_zbuffer(
            vertices_camera, faces, raster_intrinsic, height, width
        )
    all_mesh_z = np.min(np.stack(list(zbuffers.values())), axis=0)

    proposals, diagnostics = {}, {}
    for pre_identity, post_identity in pairs:
        human = cache["humans"][-1][post_identity]
        local_root = local_roots[post_identity]
        local_root_range = float(np.linalg.norm(local_root))
        root_ray_camera = local_root / max(local_root_range, 1e-8)
        observation = observed_root_ray_residual(
            zbuffers[post_identity],
            all_mesh_z,
            raster_intrinsic,
            da3["depth"][1],
            da3["intrinsics"][1],
            da3["confidence"][1],
            depth_scale,
            root_ray_camera,
            args,
        )

        root = v1.transform_points(b0, np.asarray(human["root"])[None])[0]
        joints = v1.transform_points(b0, np.asarray(human["joints"]))
        vertices = v1.transform_points(b0, np.asarray(human["vertices"]))
        camera_center = final_camera[:3, 3]
        ray_vector = root - camera_center
        current_range = float(np.linalg.norm(ray_vector))
        ray_world = ray_vector / max(current_range, 1e-8)
        raw_delta = float(observation.get("root_ray_residual_median_m", 0.0))
        delta = (
            float(np.clip(raw_delta, -args.cap_m, args.cap_m))
            if observation["accepted"]
            else 0.0
        )
        proposals[post_identity] = {
            "base": (root, joints, vertices, 1.0),
            "corrected": v1.apply_ray_change(
                root, joints, vertices, camera_center, ray_world, delta, "translation"
            ),
            "ray": ray_world,
            "camera_center": camera_center,
        }
        mesh_projection = observation.get(
            "predicted_mesh_root_ray_projection_median_m"
        )
        diagnostics[post_identity] = {
            "pre_memory_identity": pre_identity,
            "post_detection_identity": post_identity,
            "depth_scale_from_frozen_camera_baseline": depth_scale,
            "b0_baseline_m": frozen_baseline,
            "da3_baseline_units": da3_baseline,
            "predicted_root_range_m": current_range,
            "camera_local_root_range_m": local_root_range,
            "root_range_frame_consistency_error_m": abs(
                current_range - local_root_range
            ),
            "surface_to_root_offset_on_root_ray_m": (
                local_root_range - float(mesh_projection)
                if mesh_projection is not None
                else None
            ),
            "raw_depth_residual_m": raw_delta,
            "applied_depth_residual_m": delta,
            **observation,
        }

    if not np.array_equal(final_camera, camera_snapshot):
        raise AssertionError("Mesh-depth-v2 proposal mutated frozen camera")
    return {
        "b0": b0,
        "frozen_camera": final_camera,
        "camera_snapshot": camera_snapshot,
        "people": proposals,
    }, diagnostics, mapping


def evaluate_frozen(cache: dict, frozen: dict) -> dict:
    """Evaluation-only path; GT is first accessed here."""
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64))
    target_humans = cache["gt"]["post_humans"]
    methods = {name: [] for name in METHODS}
    diagnostics = []
    for identity, proposal in frozen["people"].items():
        if identity not in target_humans:
            continue
        target = {
            key: v1.transform_points(gauge, np.asarray(target_humans[identity][key]))
            for key in ("root", "joints", "vertices")
        }
        base = proposal["base"]
        methods["b0"].append({"identity": identity, **v1.point_errors(base, target)})
        methods["mesh_depth_v2_translation_cap030"].append(
            {"identity": identity, **v1.point_errors(proposal["corrected"], target)}
        )
        root, joints, vertices = base[:3]
        ray = proposal["ray"]
        oracle_delta = float(np.dot(target["root"] - root, ray))
        oracle = v1.apply_ray_change(
            root,
            joints,
            vertices,
            proposal["camera_center"],
            ray,
            oracle_delta,
            "translation",
        )
        methods["oracle_gt_ray_translation"].append(
            {"identity": identity, **v1.point_errors(oracle, target)}
        )
        diagnostics.append(
            {"identity": identity, "oracle_depth_residual_m": oracle_delta}
        )
    return {"methods": methods, "gt_ray_diagnostics": diagnostics}


def summarize(cases: list[dict]) -> dict:
    output = {
        "case_count": len(cases),
        "person_count": sum(
            len(case["evaluation"]["methods"]["b0"]) for case in cases
        ),
        "accepted_person_count": sum(
            sum(row["accepted"] for row in case["candidate_diagnostics"].values())
            for case in cases
        ),
        "camera_bit_exact_all": all(case["camera_bit_exact"] for case in cases),
        "mapping_ok_all": all(
            case["mapping_diagnostic"]["status"] == "ok" for case in cases
        ),
        "methods": {},
    }
    baseline = [
        row for case in cases for row in case["evaluation"]["methods"]["b0"]
    ]
    for method in METHODS:
        rows = [
            row for case in cases for row in case["evaluation"]["methods"][method]
        ]
        values = {
            metric: v1.finite_stats([row[metric] for row in rows])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        deltas = [
            row["root_error_m"] - base["root_error_m"]
            for row, base in zip(rows, baseline)
        ]
        values["root_mean_delta_m"] = (
            float(np.mean(deltas)) if deltas else float("nan")
        )
        values["root_improvement_rate"] = (
            float(np.mean(np.asarray(deltas) < 0)) if deltas else float("nan")
        )
        output["methods"][method] = values

    pairs = []
    for case in cases:
        oracle = {
            row["identity"]: row["oracle_depth_residual_m"]
            for row in case["evaluation"]["gt_ray_diagnostics"]
        }
        for identity, row in case["candidate_diagnostics"].items():
            if row["accepted"] and identity in oracle:
                pairs.append([row["raw_depth_residual_m"], oracle[identity]])
    pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    output["mesh_depth_v2_vs_gt_ray"] = {
        "accepted_pair_count": len(pairs),
        "sign_agreement_rate": (
            float(np.mean(np.sign(pairs[:, 0]) == np.sign(pairs[:, 1])))
            if len(pairs)
            else float("nan")
        ),
        "pearson_correlation": (
            float(np.corrcoef(pairs.T)[0, 1]) if len(pairs) >= 2 else float("nan")
        ),
        "pairs_predicted_oracle_m": pairs,
    }
    return output


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# B0 + DA3 Person Mesh-Depth V2 — Strict Root-Ray 3-Cut Dev Probe",
        "",
        f"Cases/people: `{summary['case_count']}/{summary['person_count']}`; accepted: "
        f"`{summary['accepted_person_count']}`; camera bit-exact: "
        f"`{summary['camera_bit_exact_all']}`; mapping valid: "
        f"`{summary['mapping_ok_all']}`.",
        "",
        "| Method | Root | Joint | Vertex | Root delta | Improve |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['root_error_m']['mean']:.4f} | "
            f"{row['joint_error_m']['mean']:.4f} | "
            f"{row['vertex_error_m']['mean']:.4f} | "
            f"{row['root_mean_delta_m']:+.4f} | "
            f"{row['root_improvement_rate']:.1%} |"
        )
    cue = summary["mesh_depth_v2_vs_gt_ray"]
    lines.extend(
        [
            "",
            "Candidate: same-pixel Human3R mesh and scaled DA3 z-depth are "
            "unprojected with their own K; each residual is "
            "`dot(x_DA3-x_mesh, root_ray_camera)`. Only rigid root-ray "
            "translation is allowed.",
            "",
            "Precision sign gate: `abs(median) > 2 * MAD`, in addition to "
            "the unchanged pixel/scale/absolute-MAD gates. GT is evaluator-only.",
            "",
            f"Accepted residual vs GT-ray: sign agreement "
            f"`{cue['sign_agreement_rate']:.1%}`, Pearson correlation "
            f"`{cue['pearson_correlation']:.3f}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not (args.model_path / "model.safetensors").is_file():
        raise FileNotFoundError(args.model_path / "model.safetensors")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        v1.SEQUENCE_INPUTS["three"]["report"].read_text(encoding="utf-8")
    )
    selected = source["cases"][: int(args.max_cases)]
    faces = v1.load_faces()
    model = v1.DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    reader = v1.FrameReader(args, "three")
    cases, failures = [], []
    try:
        for index, report_case in enumerate(selected, start=1):
            key = report_case["case"]["key"]
            started = time.perf_counter()
            try:
                cache = torch.load(
                    v1.SEQUENCE_INPUTS["three"]["cache"] / f"{key}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                first = reader.read(
                    int(report_case["case"]["source_camera"]),
                    int(report_case["case"]["pre_frames"][-1]),
                )[..., ::-1].copy()
                second = reader.read(
                    int(report_case["case"]["target_camera"]),
                    int(report_case["case"]["post_frame"]),
                )[..., ::-1].copy()
                da3, da3_seconds = v1.run_da3(
                    model, first, second, args.process_res
                )
                frozen, candidate_diagnostics, mapping = build_candidates(
                    report_case, cache, da3, faces, args
                )
                evaluation = evaluate_frozen(cache, frozen)
                case = {
                    "status": "ok",
                    "case": report_case["case"],
                    "camera_bit_exact": bool(
                        np.array_equal(
                            frozen["frozen_camera"], frozen["camera_snapshot"]
                        )
                    ),
                    "mapping_diagnostic": mapping,
                    "frozen_b0": frozen["b0"],
                    "frozen_camera": frozen["frozen_camera"],
                    "candidate_diagnostics": candidate_diagnostics,
                    "evaluation": evaluation,
                    "da3_seconds": da3_seconds,
                    "wall_seconds": time.perf_counter() - started,
                }
                cases.append(case)
                print(
                    f"[{index}/{len(selected)}] {key} "
                    f"accepted={sum(row['accepted'] for row in candidate_diagnostics.values())} "
                    f"camera_exact={case['camera_bit_exact']} "
                    f"mapping={mapping['status']} seconds={case['wall_seconds']:.2f}",
                    flush=True,
                )
            except Exception as error:
                case = {
                    "status": "failed",
                    "case_key": key,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(case)
            (cases_dir / f"{key}.json").write_text(
                json.dumps(
                    v1.jsonable(case), indent=2, ensure_ascii=False, allow_nan=True
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        reader.close()

    report = {
        "experiment": "v14_b0_da3_person_mesh_depth_v2_three_dev",
        "protocol": {
            "camera": "frozen learned B0, bit-exact for all methods",
            "candidate": "same-pixel mesh/DA3 camera-3D strict root-ray projection",
            "residual": "dot(x_DA3_scaled - x_mesh, root_ray_camera)",
            "human_update": "capped rigid translation along current root ray only",
            "gt_candidate_or_gate_usage": False,
            "gt_usage": "evaluation and GT-ray oracle only after candidate freeze",
        },
        "parameters": {
            "max_cases": args.max_cases,
            "process_res": args.process_res,
            "cap_m": args.cap_m,
            "min_pixels": args.min_pixels,
            "max_residual_mad_m": args.max_residual_mad_m,
            "min_median_to_mad": args.min_median_to_mad,
            "erode_iterations": args.erode_iterations,
        },
        "summary": summarize(cases),
        "failures": failures,
        "cases": cases,
    }
    json_path = args.output_dir / "v14_b0_da3_person_mesh_depth_v2.json"
    md_path = args.output_dir / "v14_b0_da3_person_mesh_depth_v2.md"
    json_path.write_text(
        json.dumps(v1.jsonable(report), indent=2, ensure_ascii=False, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    md_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f">> {json_path}", flush=True)


if __name__ == "__main__":
    main()
