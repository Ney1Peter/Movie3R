#!/usr/bin/env python3
"""Person mesh z-buffer versus DA3 depth with a bit-exact frozen B0 camera.

Each predicted SMPL-X mesh is triangle-rasterized into the recovered Human3R
image plane, then mapped exactly to DA3's resized full-frame pixel plane.  On
the same visible mesh pixels, DA3 surface range replaces predicted mesh surface
range while the predicted surface-to-root offset is retained.  The only output
change is a capped rigid translation of that person along its current root ray.

GT camera/body fields are accessed only after all candidates and gates freeze.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset/Depth-Anything-3"
for path in (REPO_ROOT, REPO_ROOT / "src", DA3_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from dust3r.utils.geometry import get_camera_parameters  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_da3_person_pointmap import (  # noqa: E402
    apply_ray_change,
    auto_identity_pairs,
    finite_stats,
    jsonable,
    point_errors,
    run_da3,
    transform_points,
)
from versions.v14.probe_b0_sift_epipolar import FrameReader  # noqa: E402


DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_MODEL = DA3_ROOT / "checkpoints/DAE-base"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_da3_person_mesh_depth"
)
METHODS = ("b0", "mesh_depth_translation_cap030", "oracle_gt_ray_translation")
HUMAN3R_SIZE = 512
MHMR_SIZE = 896


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
    parser.add_argument("--erode_iterations", type=int, default=1)
    return parser.parse_args()


def load_faces() -> np.ndarray:
    with (REPO_ROOT / "src/models/smplx/SMPLX_NEUTRAL.pkl").open("rb") as handle:
        faces = pickle.load(handle, encoding="latin1")["f"]
    faces = np.asarray(faces, dtype=np.int32)
    if faces.shape != (20908, 3):
        raise ValueError(f"Unexpected SMPL-X faces: {faces.shape}")
    return faces


def recovered_human3r_intrinsics() -> np.ndarray:
    intrinsic = get_camera_parameters(MHMR_SIZE, device="cpu")[0].numpy().astype(np.float64)
    padding = 0.5 * (MHMR_SIZE - HUMAN3R_SIZE)
    intrinsic[0, 2] -= padding
    intrinsic[1, 2] -= padding
    return intrinsic


def transform_intrinsics_pixel_centers(
    intrinsic: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scale_x, scale_y = target_w / source_w, target_h / source_h
    output = np.asarray(intrinsic, dtype=np.float64).copy()
    output[0, 0] *= scale_x
    output[1, 1] *= scale_y
    output[0, 2] = (output[0, 2] + 0.5) * scale_x - 0.5
    output[1, 2] = (output[1, 2] + 0.5) * scale_y - 0.5
    return output


def camera_vertices(vertices_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    world_to_camera = np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    return transform_points(world_to_camera, np.asarray(vertices_world, dtype=np.float64))


def project(vertices_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    normalized = vertices_camera[:, :2] / vertices_camera[:, 2:3]
    return normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]


def clipped_bbox(points: np.ndarray, size: int) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    bbox = np.r_[points[finite].min(axis=0), points[finite].max(axis=0)]
    bbox[:2] = np.maximum(bbox[:2], 0.0)
    bbox[2:] = np.minimum(bbox[2:], size - 1.0)
    return bbox


def mapping_diagnostic(
    cache: dict, human3r_intrinsic: np.ndarray, da3_hw: tuple[int, int]
) -> dict:
    residuals = []
    per_person = []
    for cache_index in (-2, -1):
        pose = np.asarray(cache["poses"][cache_index], dtype=np.float64)
        for identity, human in cache["humans"][cache_index].items():
            vertices = camera_vertices(human["vertices"], pose)
            projected = project(vertices, human3r_intrinsic)
            recovered = clipped_bbox(projected, HUMAN3R_SIZE)
            expected = np.asarray(human["bbox"], dtype=np.float64)
            residual = float(np.max(np.abs(recovered - expected)))
            residuals.append(residual)
            per_person.append(
                {
                    "view": "pre" if cache_index == -2 else "post",
                    "identity": identity,
                    "max_bbox_coordinate_residual_px": residual,
                }
            )
    target_h, target_w = da3_hw
    return {
        "status": "ok" if max(residuals, default=float("inf")) < 1e-3 else "failed",
        "human3r_image_hw": [HUMAN3R_SIZE, HUMAN3R_SIZE],
        "da3_depth_hw": [target_h, target_w],
        "resize": "full_square_pixel_center_resize_no_crop",
        "scale_xy": [target_w / HUMAN3R_SIZE, target_h / HUMAN3R_SIZE],
        "max_bbox_coordinate_residual_px": max(residuals, default=float("nan")),
        "per_person": per_person,
    }


def rasterize_zbuffer(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Perspective-correct triangle z-buffer at integer pixel centers."""
    uv = project(vertices_camera, intrinsic)
    z_vertex = np.asarray(vertices_camera[:, 2], dtype=np.float64)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    for face in faces:
        z = z_vertex[face]
        if not np.isfinite(z).all() or float(z.min()) <= 0.03:
            continue
        triangle = uv[face]
        if not np.isfinite(triangle).all():
            continue
        x0 = max(int(math.floor(float(triangle[:, 0].min()))), 0)
        x1 = min(int(math.ceil(float(triangle[:, 0].max()))), width - 1)
        y0 = max(int(math.floor(float(triangle[:, 1].min()))), 0)
        y1 = min(int(math.ceil(float(triangle[:, 1].max()))), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = triangle[0]
        bx, by = triangle[1]
        cx, cy = triangle[2]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(float(denominator)) < 1e-10:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        first = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denominator
        second = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denominator
        third = 1.0 - first - second
        inside = (first >= -1e-7) & (second >= -1e-7) & (third >= -1e-7)
        if not inside.any():
            continue
        inverse_z = first / z[0] + second / z[1] + third / z[2]
        triangle_z = np.divide(
            1.0, inverse_z, out=np.full_like(inverse_z, np.inf), where=inverse_z > 0
        )
        view = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (triangle_z < view)
        view[update] = triangle_z[update].astype(np.float32)
    return zbuffer


def range_factor(intrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = (xx - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) / intrinsic[1, 1]
    return np.sqrt(x * x + y * y + 1.0)


def observed_surface_residual(
    mesh_z: np.ndarray,
    all_mesh_z: np.ndarray,
    human3r_factor: np.ndarray,
    da3_range: np.ndarray,
    da3_confidence: np.ndarray,
    depth_scale: float,
    args: argparse.Namespace,
) -> dict:
    finite = np.isfinite(mesh_z)
    visible = finite & (mesh_z <= all_mesh_z + 1e-4)
    if args.erode_iterations:
        visible = cv2.erode(
            visible.astype(np.uint8), np.ones((3, 3), np.uint8),
            iterations=int(args.erode_iterations),
        ).astype(bool)
    valid = (
        visible
        & np.isfinite(da3_range)
        & (da3_range > 0.02)
        & np.isfinite(da3_confidence)
    )
    if int(valid.sum()) < int(args.min_pixels):
        return {
            "accepted": False,
            "reason": "too_few_same_surface_pixels",
            "silhouette_pixels": int(finite.sum()),
            "visible_pixels": int(visible.sum()),
            "valid_pixels": int(valid.sum()),
        }
    confidence_threshold = float(np.percentile(da3_confidence[valid], 30.0))
    valid &= da3_confidence >= confidence_threshold
    predicted_surface = mesh_z[valid].astype(np.float64) * human3r_factor[valid]
    observed_surface = da3_range[valid].astype(np.float64) * float(depth_scale)
    residual = observed_surface - predicted_surface
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    accepted = bool(
        len(residual) >= int(args.min_pixels)
        and mad <= float(args.max_residual_mad_m)
        and 0.05 < float(depth_scale) < 20.0
    )
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "surface_residual_dispersion_gate",
        "silhouette_pixels": int(finite.sum()),
        "visible_pixels": int(visible.sum()),
        "valid_pixels": int(len(residual)),
        "predicted_mesh_surface_range_median_m": float(np.median(predicted_surface)),
        "scaled_da3_surface_range_median_m": float(np.median(observed_surface)),
        "surface_residual_median_m": median,
        "surface_residual_mad_m": mad,
        "confidence_mean": float(np.mean(da3_confidence[valid])),
    }


def build_candidates(
    report_case: dict,
    cache: dict,
    da3: dict,
    faces: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict]:
    """GT-free mesh/depth proposal. No cache['gt'] access is permitted here."""
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    final_camera = b0 @ post_pose
    camera_snapshot = final_camera.copy()
    height, width = da3["depth"][1].shape
    human3r_intrinsic = recovered_human3r_intrinsics()
    mapping = mapping_diagnostic(cache, human3r_intrinsic, (height, width))
    if mapping["status"] != "ok":
        raise RuntimeError(f"Unreliable Human3R→DA3 mapping: {mapping}")
    raster_intrinsic = transform_intrinsics_pixel_centers(
        human3r_intrinsic, (HUMAN3R_SIZE, HUMAN3R_SIZE), (height, width)
    )
    da3_c2w = np.linalg.inv(da3["extrinsics"])
    da3_baseline = float(np.linalg.norm(da3_c2w[1, :3, 3] - da3_c2w[0, :3, 3]))
    frozen_baseline = float(np.linalg.norm(final_camera[:3, 3] - pre_pose[:3, 3]))
    depth_scale = frozen_baseline / max(da3_baseline, 1e-8)
    da3_factor = range_factor(da3["intrinsics"][1], height, width)
    da3_range = da3["depth"][1].astype(np.float64) * da3_factor
    human3r_factor = range_factor(raster_intrinsic, height, width)

    pairs = auto_identity_pairs(report_case, cache)
    post_identities = tuple(dict.fromkeys(post for _, post in pairs))
    zbuffers = {}
    for identity in post_identities:
        human = cache["humans"][-1][identity]
        vertices_camera = camera_vertices(human["vertices"], post_pose)
        zbuffers[identity] = rasterize_zbuffer(
            vertices_camera, faces, raster_intrinsic, height, width
        )
    all_mesh_z = np.min(np.stack(list(zbuffers.values())), axis=0)
    proposals, diagnostics = {}, {}
    for pre_identity, post_identity in pairs:
        human = cache["humans"][-1][post_identity]
        observation = observed_surface_residual(
            zbuffers[post_identity], all_mesh_z, human3r_factor,
            da3_range, da3["confidence"][1], depth_scale, args,
        )
        root = transform_points(b0, np.asarray(human["root"])[None])[0]
        joints = transform_points(b0, np.asarray(human["joints"]))
        vertices = transform_points(b0, np.asarray(human["vertices"]))
        camera_center = final_camera[:3, 3]
        ray_vector = root - camera_center
        ray = ray_vector / max(float(np.linalg.norm(ray_vector)), 1e-8)
        raw_delta = float(observation.get("surface_residual_median_m", 0.0))
        delta = (
            float(np.clip(raw_delta, -args.cap_m, args.cap_m))
            if observation["accepted"] else 0.0
        )
        proposals[post_identity] = {
            "base": (root, joints, vertices, 1.0),
            "corrected": apply_ray_change(
                root, joints, vertices, camera_center, ray, delta, "translation"
            ),
            "ray": ray,
            "camera_center": camera_center,
        }
        root_range = float(np.linalg.norm(root - camera_center))
        diagnostics[post_identity] = {
            "pre_memory_identity": pre_identity,
            "post_detection_identity": post_identity,
            "depth_scale_from_frozen_camera_baseline": depth_scale,
            "predicted_root_range_m": root_range,
            "surface_to_root_offset_m": (
                root_range - observation["predicted_mesh_surface_range_median_m"]
                if "predicted_mesh_surface_range_median_m" in observation else None
            ),
            "raw_depth_residual_m": raw_delta,
            "applied_depth_residual_m": delta,
            **observation,
        }
    if not np.array_equal(final_camera, camera_snapshot):
        raise AssertionError("Mesh-depth proposal mutated frozen camera")
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
            key: transform_points(gauge, np.asarray(target_humans[identity][key]))
            for key in ("root", "joints", "vertices")
        }
        base = proposal["base"]
        methods["b0"].append({"identity": identity, **point_errors(base, target)})
        methods["mesh_depth_translation_cap030"].append(
            {"identity": identity, **point_errors(proposal["corrected"], target)}
        )
        root, joints, vertices = base[:3]
        ray = proposal["ray"]
        oracle_delta = float(np.dot(target["root"] - root, ray))
        oracle = apply_ray_change(
            root, joints, vertices, proposal["camera_center"], ray,
            oracle_delta, "translation",
        )
        methods["oracle_gt_ray_translation"].append(
            {"identity": identity, **point_errors(oracle, target)}
        )
        diagnostics.append(
            {"identity": identity, "oracle_depth_residual_m": oracle_delta}
        )
    return {"methods": methods, "gt_ray_diagnostics": diagnostics}


def summarize(cases: list[dict]) -> dict:
    output = {
        "case_count": len(cases),
        "person_count": sum(len(case["evaluation"]["methods"]["b0"]) for case in cases),
        "accepted_person_count": sum(
            sum(row["accepted"] for row in case["candidate_diagnostics"].values())
            for case in cases
        ),
        "camera_bit_exact_all": all(case["camera_bit_exact"] for case in cases),
        "mapping_ok_all": all(case["mapping_diagnostic"]["status"] == "ok" for case in cases),
        "methods": {},
    }
    baseline = [row for case in cases for row in case["evaluation"]["methods"]["b0"]]
    for method in METHODS:
        rows = [row for case in cases for row in case["evaluation"]["methods"][method]]
        values = {
            metric: finite_stats([row[metric] for row in rows])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        delta = [row["root_error_m"] - base["root_error_m"] for row, base in zip(rows, baseline)]
        values["root_mean_delta_m"] = float(np.mean(delta)) if delta else float("nan")
        values["root_improvement_rate"] = float(np.mean(np.asarray(delta) < 0)) if delta else float("nan")
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
    output["mesh_depth_vs_gt_ray"] = {
        "accepted_pair_count": len(pairs),
        "sign_agreement_rate": (
            float(np.mean(np.sign(pairs[:, 0]) == np.sign(pairs[:, 1])))
            if len(pairs) else float("nan")
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
        "# B0 + DA3 Person Mesh-Depth — 3-Cut Dev Probe",
        "",
        f"Cases/people: `{summary['case_count']}/{summary['person_count']}`; accepted: "
        f"`{summary['accepted_person_count']}`; camera bit-exact: `{summary['camera_bit_exact_all']}`; "
        f"mapping valid: `{summary['mapping_ok_all']}`.",
        "",
        "| Method | Root | Joint | Vertex | Root delta | Improve |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['root_error_m']['mean']:.4f} | "
            f"{row['joint_error_m']['mean']:.4f} | {row['vertex_error_m']['mean']:.4f} | "
            f"{row['root_mean_delta_m']:+.4f} | {row['root_improvement_rate']:.1%} |"
        )
    cue = summary["mesh_depth_vs_gt_ray"]
    lines.extend(
        [
            "",
            "Candidate: triangle z-buffer silhouette + same-pixel DA3/predicted-mesh surface "
            "residual + predicted surface-to-root offset; capped rigid root-ray translation only.",
            "",
            f"Accepted residual vs GT-ray: sign agreement `{cue['sign_agreement_rate']:.1%}`, "
            f"Pearson correlation `{cue['pearson_correlation']:.3f}`. GT-ray is evaluator-only.",
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
    source = json.loads(SEQUENCE_INPUTS["three"]["report"].read_text(encoding="utf-8"))
    selected = source["cases"][: int(args.max_cases)]
    faces = load_faces()
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    reader = FrameReader(args, "three")
    cases, failures = [], []
    try:
        for index, report_case in enumerate(selected, start=1):
            key = report_case["case"]["key"]
            started = time.perf_counter()
            try:
                cache = torch.load(
                    SEQUENCE_INPUTS["three"]["cache"] / f"{key}.pt",
                    map_location="cpu", weights_only=False,
                )
                first = reader.read(
                    int(report_case["case"]["source_camera"]),
                    int(report_case["case"]["pre_frames"][-1]),
                )[..., ::-1].copy()
                second = reader.read(
                    int(report_case["case"]["target_camera"]),
                    int(report_case["case"]["post_frame"]),
                )[..., ::-1].copy()
                da3, da3_seconds = run_da3(model, first, second, args.process_res)
                frozen, candidate_diagnostics, mapping = build_candidates(
                    report_case, cache, da3, faces, args
                )
                evaluation = evaluate_frozen(cache, frozen)
                case = {
                    "status": "ok", "case": report_case["case"],
                    "camera_bit_exact": bool(
                        np.array_equal(frozen["frozen_camera"], frozen["camera_snapshot"])
                    ),
                    "mapping_diagnostic": mapping,
                    "frozen_b0": frozen["b0"], "frozen_camera": frozen["frozen_camera"],
                    "candidate_diagnostics": candidate_diagnostics,
                    "evaluation": evaluation, "da3_seconds": da3_seconds,
                    "wall_seconds": time.perf_counter() - started,
                }
                cases.append(case)
                print(
                    f"[{index}/{len(selected)}] {key} "
                    f"accepted={sum(row['accepted'] for row in candidate_diagnostics.values())} "
                    f"camera_exact={case['camera_bit_exact']} mapping={mapping['status']} "
                    f"seconds={case['wall_seconds']:.2f}", flush=True,
                )
            except Exception as error:
                case = {
                    "status": "failed", "case_key": key, "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(case)
            (cases_dir / f"{key}.json").write_text(
                json.dumps(jsonable(case), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
                encoding="utf-8",
            )
    finally:
        reader.close()
    report = {
        "experiment": "v14_b0_da3_person_mesh_depth_three_dev",
        "protocol": {
            "camera": "frozen learned B0, bit-exact for all methods",
            "candidate": "predicted SMPL-X vertices/faces, recovered projection, triangle z-buffer, DA3 post depth/confidence, frozen predicted identity matcher",
            "human_update": "capped rigid translation along current root ray only",
            "gt_candidate_or_gate_usage": False,
            "gt_usage": "evaluation and GT-ray oracle only after candidate freeze",
            "forbidden_previous_method": "no pre-to-post bbox surface-change transfer",
        },
        "parameters": {
            "max_cases": args.max_cases, "process_res": args.process_res,
            "cap_m": args.cap_m, "min_pixels": args.min_pixels,
            "max_residual_mad_m": args.max_residual_mad_m,
            "erode_iterations": args.erode_iterations,
        },
        "summary": summarize(cases), "failures": failures, "cases": cases,
    }
    json_path = args.output_dir / "v14_b0_da3_person_mesh_depth.json"
    md_path = args.output_dir / "v14_b0_da3_person_mesh_depth.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    md_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f">> {json_path}", flush=True)


if __name__ == "__main__":
    main()
