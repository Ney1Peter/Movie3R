#!/usr/bin/env python3
"""Measure non-rigid camera/human/scene drift between V14 shadow and raw outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from types import MethodType
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import run_rollout  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    boundary_error,
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/shadow_relative_consistency"
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
REPORTS = {
    "three": REPO_ROOT
    / "output/v14/b0_identity_matching/v14_b0_identity_matching.json",
    "dance": REPO_ROOT
    / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
    "box": REPO_ROOT
    / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--point_samples", type=int, default=100000)
    parser.add_argument(
        "--sequences", nargs="+", choices=tuple(REPORTS), default=tuple(REPORTS)
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("full", "pose_only_inference"),
        default=("full", "pose_only_inference"),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Explicit case key; otherwise use the largest-view-span case per sequence.",
    )
    return parser.parse_args()


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float64)


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            key: float("nan") for key in ("count", "mean", "median", "p90", "p95", "max")
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def pointwise_distance(first: np.ndarray, second: np.ndarray, limit: int) -> dict:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    if len(first) != len(second):
        return {
            "status": "shape_mismatch",
            "first_count": len(first),
            "second_count": len(second),
        }
    valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]
    return {
        "status": "ok",
        "distance_m": distribution(np.linalg.norm(first[indices] - second[indices], axis=1)),
    }


def assigned_humans(
    args: argparse.Namespace,
    prediction: dict,
    view: dict,
    debug: dict,
    layer,
    camera: int,
    frame: int,
) -> tuple[dict[str, dict], dict]:
    humans = geometry.layer_humans(prediction, view, debug, layer)
    height, width = [int(value) for value in geometry.tensor_numpy(view["true_shape"])[0]]
    return geometry.assign_gt_identities(
        args,
        humans,
        camera_matrix(prediction),
        camera,
        frame,
        width,
        height,
    )


def human_audit(
    shadow_prediction: dict,
    raw_prediction: dict,
    shadow_humans: dict[str, dict],
    raw_humans: dict[str, dict],
    boundary: np.ndarray,
) -> dict:
    shared = tuple(sorted(set(shadow_humans) & set(raw_humans)))
    per_identity = {}
    for identity in shared:
        shadow = shadow_humans[identity]
        raw = raw_humans[identity]
        shadow_index = int(shadow["detection_index"])
        raw_index = int(raw["detection_index"])
        shadow_local_root = tensor_numpy(shadow_prediction["smpl_transl"])[
            0, shadow_index
        ]
        raw_local_root = tensor_numpy(raw_prediction["smpl_transl"])[0, raw_index]
        mapped_raw_root = geometry.transform_points(boundary, raw["root"][None])[0]
        mapped_raw_joints = geometry.transform_points(boundary, raw["joints"])
        mapped_raw_vertices = geometry.transform_points(boundary, raw["vertices"])
        joint_count = min(len(shadow["joints"]), len(mapped_raw_joints))
        vertex_count = min(len(shadow["vertices"]), len(mapped_raw_vertices))
        per_identity[identity] = {
            "shadow_detection_index": shadow_index,
            "raw_detection_index": raw_index,
            "camera_relative_root_drift_m": float(
                np.linalg.norm(shadow_local_root - raw_local_root)
            ),
            "camera_relative_depth_drift_m": float(
                abs(shadow_local_root[2] - raw_local_root[2])
            ),
            "shared_b0_world_root_residual_m": float(
                np.linalg.norm(shadow["root"] - mapped_raw_root)
            ),
            "shared_b0_world_joint_residual_m": float(
                np.mean(
                    np.linalg.norm(
                        shadow["joints"][:joint_count]
                        - mapped_raw_joints[:joint_count],
                        axis=1,
                    )
                )
            ),
            "shared_b0_world_vertex_residual_m": float(
                np.mean(
                    np.linalg.norm(
                        shadow["vertices"][:vertex_count]
                        - mapped_raw_vertices[:vertex_count],
                        axis=1,
                    )
                )
            ),
        }
    metrics = (
        "camera_relative_root_drift_m",
        "camera_relative_depth_drift_m",
        "shared_b0_world_root_residual_m",
        "shared_b0_world_joint_residual_m",
        "shared_b0_world_vertex_residual_m",
    )
    return {
        "shadow_count": len(shadow_humans),
        "raw_count": len(raw_humans),
        "shared_gt_identities": shared,
        "complete_correspondence": (
            len(shadow_humans) == len(raw_humans) == len(shared)
        ),
        "summary": {
            metric: distribution(
                np.asarray([row[metric] for row in per_identity.values()])
            )
            for metric in metrics
        },
        "per_identity": per_identity,
    }


def scene_audit(
    shadow_prediction: dict,
    raw_prediction: dict,
    boundary: np.ndarray,
    point_samples: int,
) -> dict:
    shadow_local = tensor_numpy(shadow_prediction["pts3d_in_self_view"])
    raw_local = tensor_numpy(raw_prediction["pts3d_in_self_view"])
    shadow_world = tensor_numpy(shadow_prediction["pts3d_in_other_view"])
    raw_world = tensor_numpy(raw_prediction["pts3d_in_other_view"])
    shadow_camera = camera_matrix(shadow_prediction).astype(np.float64)
    raw_camera = camera_matrix(raw_prediction).astype(np.float64)
    shadow_camera_mapped_local = geometry.transform_points(
        shadow_camera, shadow_local.reshape(-1, 3)
    ).reshape(shadow_local.shape)
    raw_camera_mapped_local = geometry.transform_points(
        raw_camera, raw_local.reshape(-1, 3)
    ).reshape(raw_local.shape)
    mapped_raw_world = geometry.transform_points(boundary, raw_world.reshape(-1, 3)).reshape(
        raw_world.shape
    )
    return {
        "camera_relative_pointmap_drift": pointwise_distance(
            shadow_local, raw_local, point_samples
        ),
        "shared_b0_world_pointmap_residual": pointwise_distance(
            shadow_world, mapped_raw_world, point_samples
        ),
        "shadow_internal_camera_pointmap_residual": pointwise_distance(
            shadow_world, shadow_camera_mapped_local, point_samples
        ),
        "raw_internal_camera_pointmap_residual": pointwise_distance(
            raw_world, raw_camera_mapped_local, point_samples
        ),
    }


def configure_variant(model, variant: str) -> None:
    from dust3r.v8_head_lora import set_lora_enabled

    if not hasattr(model, "_v14_full_head_lora_router"):
        model._v14_full_head_lora_router = model._set_v14_1_head_lora_for_event
    if variant == "full":
        model.enable_v8_human_latent_corr = bool(model.v8_human_latent_corr)
        model._set_v14_1_head_lora_for_event = model._v14_full_head_lora_router
        return

    model.enable_v8_human_latent_corr = False

    def pose_only_router(self, enabled):
        if not getattr(self, "v14_1_event_only_head_lora", False):
            return
        active = bool(enabled) and bool(getattr(self, "enable_v8_head_lora", False))
        if hasattr(self.downstream_head, "pose_head"):
            set_lora_enabled(self.downstream_head.pose_head, active)
        for attr in ("deccam", "decpose", "decshape", "decexpression"):
            if hasattr(self.downstream_head, attr):
                set_lora_enabled(getattr(self.downstream_head, attr), False)

    model._set_v14_1_head_lora_for_event = MethodType(pose_only_router, model)


def select_cases(args: argparse.Namespace) -> list[tuple[str, dict]]:
    explicit = set(args.case)
    selected = []
    for sequence in args.sequences:
        rows = json.loads(REPORTS[sequence].read_text())["cases"]
        if explicit:
            rows = [row for row in rows if row["case"]["key"] in explicit]
        else:
            rows = [max(rows, key=lambda row: float(row["camera_span_deg"]))]
        selected.extend((sequence, row) for row in rows)
    missing = explicit - {row["case"]["key"] for _, row in selected}
    if missing:
        raise KeyError(f"Cases not found in selected sequence reports: {sorted(missing)}")
    return selected


def process_case(
    model,
    layer,
    args: argparse.Namespace,
    variant: str,
    sequence: str,
    row: dict,
) -> dict:
    from dust3r.v14_outputs import boundary_from_camera_predictions

    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    case = row["case"]
    local_args = SimpleNamespace(
        data_root=args.data_root,
        output_dir=args.output_dir / sequence,
        size=args.size,
        sequence=sequence,
        device=args.device,
    )
    pre_paths = [
        geometry.extract_video_frame(
            local_args, int(case["source_camera"]), int(frame)
        )
        for frame in case["pre_frames"]
    ]
    post_path = geometry.extract_video_frame(
        local_args, int(case["target_camera"]), int(case["post_frame"])
    )
    shadow_views = set_event_indices(
        geometry.prepare_full_square_input(model, pre_paths + [post_path], local_args),
        {len(pre_paths)},
    )
    raw_views = set_event_indices(
        geometry.prepare_full_square_input(model, [post_path], local_args), set()
    )
    shadow_predictions, shadow_returned, shadow_debug, shadow_time = run_rollout(
        model, shadow_views, args.device, f"{case['key']}_shadow"
    )
    raw_predictions, raw_returned, raw_debug, raw_time = run_rollout(
        model, raw_views, args.device, f"{case['key']}_raw"
    )
    shadow_prediction = shadow_predictions[-1]
    raw_prediction = raw_predictions[0]
    boundary_tensor = boundary_from_camera_predictions(shadow_prediction, raw_prediction)
    boundary = tensor_numpy(boundary_tensor)[0]
    target_boundary = np.asarray(row["boundaries"]["gt_camera"], dtype=np.float64)
    shadow_humans, shadow_assignment = assigned_humans(
        local_args,
        shadow_prediction,
        shadow_returned[-1],
        shadow_debug[-1],
        layer,
        int(case["target_camera"]),
        int(case["post_frame"]),
    )
    raw_humans, raw_assignment = assigned_humans(
        local_args,
        raw_prediction,
        raw_returned[0],
        raw_debug[0],
        layer,
        int(case["target_camera"]),
        int(case["post_frame"]),
    )
    shadow_camera = camera_matrix(shadow_prediction).astype(np.float64)
    raw_camera = camera_matrix(raw_prediction).astype(np.float64)
    return {
        "sequence": sequence,
        "variant": variant,
        "case": case,
        "camera_span_deg": row["camera_span_deg"],
        "timing_seconds": {"shadow": shadow_time, "raw": raw_time},
        "boundary": boundary,
        "camera_identity_residual": boundary_error(
            shadow_camera, boundary @ raw_camera
        ),
        "b0_gt_camera_error": boundary_error(boundary, target_boundary),
        "human": human_audit(
            shadow_prediction, raw_prediction, shadow_humans, raw_humans, boundary
        ),
        "scene": scene_audit(
            shadow_prediction, raw_prediction, boundary, int(args.point_samples)
        ),
        "gt_assignment": {"shadow": shadow_assignment, "raw": raw_assignment},
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.smpl_layer import SMPL_Layer

    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    selected = select_cases(args)
    variants = {}
    started = time.perf_counter()
    for variant in args.variants:
        configure_variant(model, variant)
        variants[variant] = []
        for sequence, row in selected:
            variants[variant].append(
                process_case(model, layer, args, variant, sequence, row)
            )
            torch.cuda.empty_cache()
    report = {
        "experiment": "v14_shadow_relative_consistency",
        "model_path": str(args.model_path.resolve()),
        "model_flags": flags,
        "protocol": {
            "shadow": "pre-cut five-frame history plus first post-cut event frame",
            "raw": "same first post-cut frame from fresh state with event correction off",
            "boundary": "C_shadow @ inverse(C_raw)",
            "identity": "GT assignment used only to compare the same predicted person",
            "point_samples_per_case": int(args.point_samples),
        },
        "runtime_seconds": time.perf_counter() - started,
        "variant_note": {
            "full": "Checkpoint's trained camera and human correction paths.",
            "pose_only_inference": (
                "Diagnostic only: human latent correction and human-head LoRA are disabled "
                "at inference. The checkpoint was jointly trained, so this is not a fair "
                "camera-only training comparison."
            ),
        },
        "variants": variants,
    }
    path = args.output_dir / "v14_shadow_relative_consistency.json"
    path.write_text(json.dumps(serializable(report), indent=2, allow_nan=True) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
