#!/usr/bin/env python3
"""Minimal CUT3R virtual-view person-depth observability probe.

The deployable phase runs only the first frozen MultiHuman ``three`` cut:

1. predict the raw first-post camera/person meshes without GT;
2. left-multiply the already frozen learned B0;
3. replay a short pre-shot RGB context plus one ray-only query through the
   full Human3R forward path;
4. assert that the query leaves recurrent state and pose memory bit-exact;
5. freeze virtual pointmaps/confidence/person-region evidence to disk.

Only after the frozen marker exists is GT loaded for identity assignment and
for evaluating the direction of virtual root-ray evidence.  No correction is
generated or applied by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.boundary_human3r_reset_support import build_smpl_models  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402
from versions.v14.probe_b0_da3_person_mesh_depth import (  # noqa: E402
    load_faces,
    rasterize_zbuffer,
)
from versions.v14.probe_b0_identity_matching import strict_cache  # noqa: E402
from versions.v14.probe_v14_internal_root_depth import decode_local_humans  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_REPORT = (
    REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json"
)
DEFAULT_CACHE = REPO_ROOT / "output/v20_phase1_gt_id_multihuman_consensus/case_cache"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/cut3r_virtual_person_depth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history_frames", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def finite_stats(value: np.ndarray) -> dict:
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "finite_count": 0,
            "finite_fraction": 0.0,
            "minimum": float("nan"),
            "q10": float("nan"),
            "median": float("nan"),
            "q90": float("nan"),
            "maximum": float("nan"),
            "mean": float("nan"),
        }
    return {
        "finite_count": int(len(finite)),
        "finite_fraction": float(len(finite) / max(array.size, 1)),
        "minimum": float(np.min(finite)),
        "q10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "q90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def first_three_case(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError(f"No cases in {path}")
    row = cases[0]
    if str(row["case"]["key"]) != "three_t0500_c0_c1_k0":
        raise ValueError(
            "Frozen first-cut expectation changed: "
            f"found {row['case']['key']!r}"
        )
    return row


def recovered_intrinsic(view: dict, height: int, width: int) -> np.ndarray:
    intrinsic = np.asarray(tensor_numpy(view["K_mhmr"])[0], dtype=np.float64).copy()
    padded_height = int(view["img_mhmr"].shape[-2])
    padded_width = int(view["img_mhmr"].shape[-1])
    intrinsic[0, 2] -= 0.5 * (padded_width - width)
    intrinsic[1, 2] -= 0.5 * (padded_height - height)
    return intrinsic


def tensor_state_comparison(before: tuple, after: tuple) -> dict:
    names = ("state_feat", "state_pos", "init_state_feat", "mem", "init_mem")
    rows = {}
    all_exact = True
    for index, name in enumerate(names):
        first = before[index]
        second = after[index]
        if first is None or second is None:
            exact = first is None and second is None
            rows[name] = {"bit_exact": exact, "both_none": exact}
            all_exact &= exact
            continue
        exact = bool(torch.equal(first, second))
        finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
        difference = (first.detach().float() - second.detach().float()).abs()
        rows[name] = {
            "bit_exact": exact,
            "finite": finite,
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "max_abs_difference": float(difference.max().item()),
        }
        all_exact &= exact and finite
    rows["all_bit_exact_and_finite"] = bool(all_exact)
    rows["required_state_and_mem_bit_exact"] = bool(
        rows["state_feat"]["bit_exact"] and rows["mem"]["bit_exact"]
    )
    return rows


def make_full_views(
    model,
    pre_paths: list[Path],
    ray_map: np.ndarray,
    intrinsic: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    pre_views = geometry.prepare_full_square_input(model, pre_paths, args)
    pre_views = set_event_indices(pre_views, set())
    height, width = map(int, ray_map.shape[:2])
    for view in pre_views:
        batch = int(view["img"].shape[0])
        view["ray_map"] = torch.zeros(
            batch, height, width, 6, dtype=view["img"].dtype
        )
        view["ray_mask"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
        for key in ("update", "update_state", "update_mem", "update_v8_history"):
            view[key] = torch.ones_like(view["img_mask"], dtype=torch.bool)

    reference = pre_views[0]
    query = {
        "img": torch.zeros_like(reference["img"]),
        "img_mhmr": torch.zeros_like(reference["img_mhmr"]),
        "K_mhmr": reference["K_mhmr"].clone(),
        "true_shape": torch.tensor([[height, width]], dtype=torch.int32),
        "ray_map": torch.from_numpy(np.asarray(ray_map, dtype=np.float32)).unsqueeze(0),
        "img_mask": torch.tensor([False], dtype=torch.bool),
        "ray_mask": torch.tensor([True], dtype=torch.bool),
        "update": torch.tensor([False], dtype=torch.bool),
        "update_state": torch.tensor([False], dtype=torch.bool),
        "update_mem": torch.tensor([False], dtype=torch.bool),
        "update_v8_history": torch.tensor([False], dtype=torch.bool),
        "reset": torch.tensor([False], dtype=torch.bool),
        "shot_label": torch.tensor([0.0], dtype=reference["img"].dtype),
        "idx": len(pre_views),
        "instance": f"virtual_query_{len(pre_views)}",
        # Placeholder only: the actual camera condition is ray_map.
        "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
    }
    query["K_mhmr"][0] = torch.from_numpy(intrinsic).to(
        dtype=query["K_mhmr"].dtype
    )
    return pre_views + [query]


def reconstruct_zbuffer_points(zbuffer: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = zbuffer.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    points = np.empty((height, width, 3), dtype=np.float64)
    points[..., 2] = zbuffer
    points[..., 0] = (xx - intrinsic[0, 2]) / intrinsic[0, 0] * zbuffer
    points[..., 1] = (yy - intrinsic[1, 2]) / intrinsic[1, 1] * zbuffer
    return points


def person_observability(
    local_humans: list[dict],
    points_self: np.ndarray,
    confidence_self: np.ndarray,
    intrinsic: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[dict], list[np.ndarray]]:
    height, width = points_self.shape[:2]
    zbuffers = [
        rasterize_zbuffer(human["vertices"], faces, intrinsic, height, width)
        for human in local_humans
    ]
    if not zbuffers:
        return [], []
    all_z = np.min(np.stack(zbuffers, axis=0), axis=0)
    point_valid = np.isfinite(points_self).all(axis=2)
    point_valid &= (points_self[..., 2] > 0.03) & (points_self[..., 2] < 100.0)
    point_valid &= np.isfinite(confidence_self)
    rows = []
    masks = []
    for human, zbuffer in zip(local_humans, zbuffers):
        silhouette = np.isfinite(zbuffer)
        visible = silhouette & (zbuffer <= all_z + 1e-4)
        valid = visible & point_valid
        masks.append(visible)
        root = np.asarray(human["root"], dtype=np.float64)
        root_length = float(np.linalg.norm(root))
        root_ray = root / max(root_length, 1e-8)
        mesh_points = reconstruct_zbuffer_points(zbuffer, intrinsic)
        same_pixel_residual = (
            np.sum(points_self[valid] * root_ray[None], axis=1)
            - np.sum(mesh_points[valid] * root_ray[None], axis=1)
            if int(valid.sum())
            else np.empty((0,), dtype=np.float64)
        )
        evidence = (
            float(np.median(same_pixel_residual))
            if len(same_pixel_residual)
            else float("nan")
        )
        residual_mad = (
            float(np.median(np.abs(same_pixel_residual - evidence)))
            if len(same_pixel_residual)
            else float("nan")
        )
        rows.append(
            {
                "detection_index": int(human["detection_index"]),
                "detection_score": float(human["score"]),
                "predicted_root_camera": root,
                "predicted_root_ray_length_m": root_length,
                "predicted_root_ray": root_ray,
                "silhouette_pixels": int(silhouette.sum()),
                "visible_mesh_pixels": int(visible.sum()),
                "virtual_valid_pixels": int(valid.sum()),
                "virtual_valid_coverage": float(valid.sum() / max(visible.sum(), 1)),
                "virtual_confidence": finite_stats(confidence_self[valid]),
                "virtual_z": finite_stats(points_self[..., 2][valid]),
                "predicted_mesh_z": finite_stats(zbuffer[valid]),
                # Frozen observability scalar only; it is never applied to the body.
                "virtual_minus_predicted_surface_root_ray_median_m": evidence,
                "virtual_minus_predicted_surface_root_ray_mad_m": residual_mad,
                "candidate_generated": False,
                "human_modified": False,
            }
        )
    return rows, masks


def run_deployable_phase(model, layer, args: argparse.Namespace, report_row: dict) -> dict:
    from dust3r.datasets.base.base_multiview_dataset import get_ray_map
    from dust3r.inference import inference

    case = report_row["case"]
    pre_frames = [int(value) for value in case["pre_frames"]][
        -int(args.history_frames) :
    ]
    source_camera = int(case["source_camera"])
    target_camera = int(case["target_camera"])
    pre_paths = [
        geometry.extract_video_frame(args, source_camera, frame)
        for frame in pre_frames
    ]
    post_path = geometry.extract_video_frame(
        args, target_camera, int(case["post_frame"])
    )

    # The raw post camera/person mesh is deployable RGB-only evidence.
    raw_views = set_event_indices(
        geometry.prepare_full_square_input(model, [post_path], args), set()
    )
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        raw_predictions, raw_returned_views, raw_debug = model.forward_recurrent_lighter(
            raw_views,
            str(args.device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    raw_prediction = raw_predictions[0]
    raw_view = raw_returned_views[0]
    raw_camera = camera_matrix(raw_prediction).astype(np.float64)
    b0 = np.asarray(report_row["boundaries"]["learned_b0"], dtype=np.float64)
    frozen_post_camera = b0 @ raw_camera
    local_humans = decode_local_humans(
        raw_prediction, raw_view, raw_debug[0], layer
    )
    world_humans = geometry.layer_humans(
        raw_prediction, raw_view, raw_debug[0], layer
    )

    height, width = [int(value) for value in tensor_numpy(raw_view["true_shape"])[0]]
    intrinsic = recovered_intrinsic(raw_view, height, width)
    ray_map = get_ray_map(
        np.eye(4, dtype=np.float64),
        frozen_post_camera,
        intrinsic,
        height,
        width,
    ).astype(np.float32)
    full_views = make_full_views(model, pre_paths, ray_map, intrinsic, args)
    query_flags = {
        key: bool(full_views[-1][key][0].item())
        for key in (
            "img_mask",
            "ray_mask",
            "update",
            "update_state",
            "update_mem",
            "update_v8_history",
            "reset",
        )
    }
    started = time.perf_counter()
    result, state_args = inference(full_views, model, str(args.device), verbose=True)
    elapsed = time.perf_counter() - started
    if len(state_args) != len(full_views) + 1:
        raise AssertionError(
            f"Expected {len(full_views)+1} state snapshots, got {len(state_args)}"
        )
    state_check = tensor_state_comparison(state_args[-2], state_args[-1])
    if not state_check["required_state_and_mem_bit_exact"]:
        raise AssertionError(f"Ray query changed state/mem: {state_check}")

    query_prediction = result["pred"][-1]
    points_self = np.asarray(
        tensor_numpy(query_prediction["pts3d_in_self_view"]), dtype=np.float32
    ).squeeze(0)
    confidence_self = np.asarray(
        tensor_numpy(query_prediction["conf_self"]), dtype=np.float32
    ).squeeze(0)
    points_world = np.asarray(
        tensor_numpy(query_prediction["pts3d_in_other_view"]), dtype=np.float32
    ).squeeze(0)
    confidence_world = np.asarray(
        tensor_numpy(query_prediction["conf"]), dtype=np.float32
    ).squeeze(0)
    expected_points = (height, width, 3)
    expected_conf = (height, width)
    if points_self.shape != expected_points or points_world.shape != expected_points:
        raise ValueError(
            f"Unexpected pointmap shapes self={points_self.shape}, world={points_world.shape}"
        )
    if confidence_self.shape != expected_conf or confidence_world.shape != expected_conf:
        raise ValueError(
            "Unexpected confidence shapes "
            f"self={confidence_self.shape}, world={confidence_world.shape}"
        )

    faces = load_faces()
    people, masks = person_observability(
        local_humans, points_self, confidence_self, intrinsic, faces
    )
    deployable = {
        "case": case,
        "history_frames": pre_frames,
        "post_image": str(post_path),
        "query_flags": query_flags,
        "model_input_count": len(full_views),
        "full_forward_seconds": elapsed,
        "b0_matrix": b0,
        "raw_post_camera": raw_camera,
        "frozen_post_b0_camera": frozen_post_camera,
        "frozen_post_b0_camera_sha256": sha256_array(frozen_post_camera),
        "query_intrinsic": intrinsic,
        "ray_map_shape": list(ray_map.shape),
        "ray_map_finite_fraction": float(np.isfinite(ray_map).mean()),
        "state_check": state_check,
        "virtual_outputs": {
            "pts3d_in_self_view_shape": list(points_self.shape),
            "pts3d_in_other_view_shape": list(points_world.shape),
            "conf_self_shape": list(confidence_self.shape),
            "conf_shape": list(confidence_world.shape),
            "self_point_finite_fraction": float(
                np.isfinite(points_self).all(axis=2).mean()
            ),
            "world_point_finite_fraction": float(
                np.isfinite(points_world).all(axis=2).mean()
            ),
            "conf_self": finite_stats(confidence_self),
            "conf_world": finite_stats(confidence_world),
            "self_depth_z": finite_stats(points_self[..., 2]),
            "world_depth_z": finite_stats(points_world[..., 2]),
        },
        "person_observability": people,
        "protocol": {
            "gt_loaded": False,
            "candidate_generated": False,
            "human_modified": False,
            "camera_modified_after_freeze": False,
            "query_path": "full model.forward replay through dust3r.inference.inference",
        },
    }
    runtime = {
        "raw_prediction": raw_prediction,
        "raw_view": raw_view,
        "raw_debug": raw_debug[0],
        "world_humans": world_humans,
        "local_humans": local_humans,
        "points_self": points_self,
        "confidence_self": confidence_self,
        "points_world": points_world,
        "confidence_world": confidence_world,
        "ray_map": ray_map,
        "masks": masks,
        "deployable": deployable,
    }
    return runtime


def freeze_deployable(runtime: dict, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "virtual_pts3d_self.npy", runtime["points_self"])
    np.save(case_dir / "virtual_conf_self.npy", runtime["confidence_self"])
    np.save(case_dir / "virtual_pts3d_world.npy", runtime["points_world"])
    np.save(case_dir / "virtual_conf_world.npy", runtime["confidence_world"])
    np.save(case_dir / "query_ray_map.npy", runtime["ray_map"])
    for index, mask in enumerate(runtime["masks"]):
        np.save(case_dir / f"person_{index:02d}_visible_mesh_mask.npy", mask)
        cv2.imwrite(
            str(case_dir / f"person_{index:02d}_visible_mesh_mask.png"),
            mask.astype(np.uint8) * 255,
        )
    deployable_path = case_dir / "deployable_frozen_no_gt.json"
    deployable_path.write_text(
        json.dumps(jsonable(runtime["deployable"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    marker = {
        "status": "frozen_before_gt",
        "deployable_json": str(deployable_path),
        "deployable_json_sha256": hashlib.sha256(deployable_path.read_bytes()).hexdigest(),
        "camera_sha256": runtime["deployable"]["frozen_post_b0_camera_sha256"],
        "gt_loaded_before_marker": False,
        "candidate_generated": False,
        "human_modified": False,
    }
    (case_dir / "FROZEN_BEFORE_GT.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def safe_correlation(first: list[float], second: list[float]) -> dict:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return {"count": int(len(x)), "pearson": float("nan"), "spearman": float("nan")}
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if np.std(x) > 1e-12 and np.std(y) > 1e-12
        else float("nan")
    )
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    spearman = (
        float(np.corrcoef(x_rank, y_rank)[0, 1])
        if np.std(x_rank) > 0 and np.std(y_rank) > 0
        else float("nan")
    )
    return {"count": int(len(x)), "pearson": pearson, "spearman": spearman}


def evaluate_after_freeze(
    runtime: dict,
    args: argparse.Namespace,
    case_dir: Path,
) -> dict:
    marker_path = case_dir / "FROZEN_BEFORE_GT.json"
    if not marker_path.is_file():
        raise RuntimeError("Refusing GT evaluation before frozen marker exists")
    case = runtime["deployable"]["case"]
    cache_path = args.cache_dir / f"{case['key']}.pt"
    cache = strict_cache(args, cache_path)
    raw_camera = np.asarray(runtime["deployable"]["raw_post_camera"], dtype=np.float64)
    height, width = runtime["points_self"].shape[:2]
    assigned, assignment = geometry.assign_gt_identities(
        args,
        runtime["world_humans"],
        raw_camera,
        int(case["target_camera"]),
        int(case["post_frame"]),
        width,
        height,
    )
    identity_by_detection = {
        int(human["detection_index"]): identity for identity, human in assigned.items()
    }
    local_by_detection = {
        int(human["detection_index"]): human for human in runtime["local_humans"]
    }
    gt_post_camera = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    rows = []
    evidence_values, gt_values = [], []
    for frozen in runtime["deployable"]["person_observability"]:
        detection = int(frozen["detection_index"])
        identity = identity_by_detection.get(detection)
        if identity is None or identity not in cache["gt"]["post_humans"]:
            rows.append(
                {
                    "detection_index": detection,
                    "status": "unassigned",
                    "frozen_evidence_m": frozen[
                        "virtual_minus_predicted_surface_root_ray_median_m"
                    ],
                }
            )
            continue
        human = local_by_detection[detection]
        predicted_root = np.asarray(human["root"], dtype=np.float64)
        root_ray = predicted_root / max(float(np.linalg.norm(predicted_root)), 1e-8)
        target_world = np.asarray(cache["gt"]["post_humans"][identity]["root"], dtype=np.float64)
        target_local = geometry.transform_points(
            np.linalg.inv(gt_post_camera), target_world[None]
        )[0]
        predicted_lambda = float(np.dot(predicted_root, root_ray))
        target_lambda = float(np.dot(target_local, root_ray))
        gt_residual = target_lambda - predicted_lambda
        evidence = float(
            frozen["virtual_minus_predicted_surface_root_ray_median_m"]
        )
        evidence_values.append(evidence)
        gt_values.append(gt_residual)
        rows.append(
            {
                "detection_index": detection,
                "identity": identity,
                "status": "evaluated",
                "frozen_evidence_m": evidence,
                "gt_root_ray_residual_m": gt_residual,
                "same_direction": bool(
                    np.isfinite(evidence)
                    and np.isfinite(gt_residual)
                    and np.sign(evidence) == np.sign(gt_residual)
                ),
                "predicted_root_camera": predicted_root,
                "gt_root_camera": target_local,
            }
        )
    valid_rows = [row for row in rows if row.get("status") == "evaluated"]
    sign_rows = [
        row
        for row in valid_rows
        if np.isfinite(row["frozen_evidence_m"])
        and np.isfinite(row["gt_root_ray_residual_m"])
    ]
    cache_raw = np.asarray(cache["poses"][-1], dtype=np.float64)
    return {
        "gt_loaded_only_after_frozen_marker": True,
        "cache_path": str(cache_path),
        "assignment": assignment,
        "people": rows,
        "direction_summary": {
            "sign_agreement_rate": float(
                np.mean([row["same_direction"] for row in sign_rows])
            )
            if sign_rows
            else float("nan"),
            "correlation": safe_correlation(evidence_values, gt_values),
            "interpretation": "positive correlation/sign agreement means virtual evidence points toward the GT root-ray residual; no candidate is applied",
        },
        "fresh_raw_vs_strict_cache": {
            "translation_m": float(
                np.linalg.norm(raw_camera[:3, 3] - cache_raw[:3, 3])
            ),
            "rotation_matrix_max_abs": float(
                np.max(np.abs(raw_camera[:3, :3] - cache_raw[:3, :3]))
            ),
        },
        "candidate_generated": False,
        "human_modified": False,
    }


def markdown(report: dict) -> str:
    deployable = report["deployable"]
    evaluation = report["evaluation"]
    outputs = deployable["virtual_outputs"]
    direction = evaluation["direction_summary"]
    lines = [
        "# CUT3R Virtual Person-Depth — First `three` Cut",
        "",
        f"Case: `{deployable['case']['key']}`. This is observability only: no root/camera candidate was generated or applied.",
        "",
        f"- Query flags: `{deployable['query_flags']}`",
        f"- State/memory bit-exact: `{deployable['state_check']['required_state_and_mem_bit_exact']}`",
        f"- Virtual self/world shapes: `{outputs['pts3d_in_self_view_shape']}` / `{outputs['pts3d_in_other_view_shape']}`",
        f"- Self/world finite rate: `{outputs['self_point_finite_fraction']:.2%}` / `{outputs['world_point_finite_fraction']:.2%}`",
        f"- Self confidence median/q90: `{outputs['conf_self']['median']:.4f}` / `{outputs['conf_self']['q90']:.4f}`",
        "",
        "## Person mesh regions",
        "",
        "| Detection | Visible mesh px | Virtual coverage | Virtual z median | Ray evidence median/MAD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for person in deployable["person_observability"]:
        lines.append(
            f"| {person['detection_index']} | {person['visible_mesh_pixels']} | "
            f"{person['virtual_valid_coverage']:.1%} | {person['virtual_z']['median']:.3f} | "
            f"{person['virtual_minus_predicted_surface_root_ray_median_m']:.3f} / "
            f"{person['virtual_minus_predicted_surface_root_ray_mad_m']:.3f} |"
        )
    corr = direction["correlation"]
    lines.extend(
        [
            "",
            "## GT-only evaluation after freeze",
            "",
            f"- Evaluated people: `{corr['count']}`",
            f"- Evidence/GT residual sign agreement: `{direction['sign_agreement_rate']:.1%}`",
            f"- Pearson/Spearman: `{corr['pearson']:.3f}` / `{corr['spearman']:.3f}`",
            "- GT was loaded only after `FROZEN_BEFORE_GT.json` was written.",
            "- These three people are an observability diagnostic, not a method result.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace, stage: dict) -> dict:
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA")
    args.sequence = "three"
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES["three"]
    report_row = first_three_case(args.report_path)
    case_dir = args.output_dir / str(report_row["case"]["key"])
    if case_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Use --overwrite for existing output: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)

    stage["name"] = "load_model"
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    flags = configure_model(model)
    _, layer = build_smpl_models(model, torch.device(args.device))

    stage["name"] = "deployable_full_forward"
    runtime = run_deployable_phase(model, layer, args, report_row)
    runtime["deployable"]["model_path"] = str(args.model_path)
    runtime["deployable"]["model_flags"] = flags

    stage["name"] = "freeze_before_gt"
    freeze_deployable(runtime, case_dir)

    stage["name"] = "gt_evaluation_after_freeze"
    evaluation = evaluate_after_freeze(runtime, args, case_dir)
    report = {
        "experiment": "v14_cut3r_virtual_person_depth_observability_first_three_cut",
        "deployable": runtime["deployable"],
        "evaluation": evaluation,
        "protocol": {
            "gt_candidate_or_gate_usage": False,
            "gt_loaded_only_after_query_freeze": True,
            "candidate_generated": False,
            "human_modified": False,
            "camera_modified_after_frozen_b0": False,
        },
    }
    (case_dir / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (case_dir / "README.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = {"name": "startup"}
    started = time.perf_counter()
    try:
        report = run(args, stage)
        print(markdown(report), flush=True)
        print(
            f">> completed in {time.perf_counter() - started:.2f}s: {args.output_dir}",
            flush=True,
        )
    except Exception as error:
        text = traceback.format_exc()
        failure = {
            "status": "failed",
            "stage": stage["name"],
            "error": repr(error),
            "traceback": text,
            "argv": sys.argv,
            "minimal_reproduction": (
                f"{Path(sys.executable)} {Path(__file__).resolve()} "
                f"--device {args.device} --overwrite"
            ),
            "model_path": str(args.model_path),
            "report_path": str(args.report_path),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.output_dir / "traceback.txt").write_text(text, encoding="utf-8")
        print(text, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
