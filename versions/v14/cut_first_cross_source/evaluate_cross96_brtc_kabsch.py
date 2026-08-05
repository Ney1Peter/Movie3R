#!/usr/bin/env python3
"""Unified cross96 first-post-cut evaluation for frozen B0/BRTC/Kabsch.

This evaluator deliberately re-runs one *cross96* forward for every event and
then applies the already-frozen person-local policies without changing their
parameters.  It therefore closes the provenance gap between the cross-source
camera result and the earlier BRTC/Kabsch evidence, which used a different B0
checkpoint.

The frozen cross96 dataset/model are configured with ``max_humans=1``.  The
only association used here is consequently the protocol-valid singleton match
``(0, 0)`` when both predictions contain exactly one person.  A missing or
non-singleton detection receives exact B0 fallback; this script must not be
used to claim multi-person or automatic-ID performance.

Runtime contract:
  shadow pre-cut state + corrected first post-cut frame -> camera B0 proposal
  clean raw reset first post-cut frame -> B0-transformed human geometry
  BRTC-LC v1 -> qualified person-local TORSO4 Kabsch candidate

The shadow recurrent state and its human mesh are never committed.  BRTC and
Kabsch only edit copied person geometry; every camera remains bit-exact B0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
THIS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from evaluate_cut_events import (  # noqa: E402
    add_mhmr_inputs,
    homogeneous,
    read_jsonl,
)
from evaluate_four_source_b0 import load_views, safe_name  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from versions.v13.gt_id_consensus import layer_humans  # noqa: E402
from versions.v14.b0_person_triangulation import (  # noqa: E402
    DEFAULT_CONFIG as BRTC_CONFIG,
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_orientation_kabsch import (  # noqa: E402
    DEFAULT_ORIENTATION_KABSCH_CONFIG,
    refine_matched_people_orientation_kabsch,
)
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    gt_head_world,
    model_batch_from_gt,
    rotation_error_deg,
    set_event_indices,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6"
    / "checkpoint-final.pth"
)
DEFAULT_RECORDS = (
    REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/eval_cross96_brtc_kabsch"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
METHODS = (
    "raw_reset",
    "shadow_event_diagnostic",
    "b0_runtime",
    "brtc_lc_v1",
    "brtc_lc_v1_torso4_kabsch_candidate",
)
HUMAN_METRICS = ("root_error_m", "joint_error_m", "vertex_error_m", "head_error_m")
CAMERA_METRICS = (
    "camera_translation_m",
    "camera_rotation_deg",
    "camera_composite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument("--max-cases-per-source", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash the checkpoint before evaluation (slow but useful for freeze provenance).",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def finite_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_event_views(record: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    loader_args = SimpleNamespace(
        data_root=args.data_root,
        resolution=(512, 288),
        resize_mode="human3r_demo",
        boundary=2,
    )
    views = load_views(record, loader_args)
    if len(views) < 3:
        raise RuntimeError(f"Expected three event views, got {len(views)}")
    return views[:3]


def transform_points(transform: np.ndarray, points: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def transform_person(transform: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
    """Apply a shared B0 to copied human geometry and orientation attributes."""

    transform = homogeneous(transform).astype(np.float64)
    output = dict(person)
    for key in ("root", "joints", "vertices"):
        if key in person:
            output[key] = transform_points(transform, person[key])
    for key in ("torso", "root_rotation"):
        if key in person:
            output[key] = transform[:3, :3] @ np.asarray(person[key], dtype=np.float64)
    return output


def gt_people_world(view: dict[str, Any], evaluation_gauge: np.ndarray, layer: SMPL_Layer) -> list[dict[str, Any]]:
    """Regenerate masked GT SMPL-X meshes in the same evaluation world gauge.

    The target's ``smplx_transl`` is standard pelvis translation, whereas the
    Human3R output ``smpl_transl`` is head-centered.  We therefore use
    ``bm_x`` directly for GT and compare its true pelvis/joints/vertices to
    geometries reconstructed by ``layer_humans``.
    """

    mask = view.get("smpl_mask")
    required = (
        "smplx_root_pose",
        "smplx_body_pose",
        "smplx_left_hand_pose",
        "smplx_right_hand_pose",
        "smplx_jaw_pose",
        "smplx_shape",
        "smplx_transl",
    )
    if mask is None or any(key not in view for key in required):
        return []
    indices = torch.nonzero(mask[0].detach().cpu().bool(), as_tuple=False).flatten()
    if not indices.numel():
        return []
    device = next(layer.parameters()).device
    params_are_world = view.get("human_params_are_world")
    are_world = bool(params_are_world is None or params_are_world[0].detach().cpu())
    camera = homogeneous(
        view.get("raw_camera_pose", view["camera_pose"])[0].detach().float().cpu().numpy()
    )
    output = []
    for index_t in indices:
        index = int(index_t)

        def parameter(key: str) -> torch.Tensor:
            return view[key][0, index].detach().float().to(device)

        with torch.no_grad():
            body = layer.bm_x(
                global_orient=parameter("smplx_root_pose").reshape(1, 3),
                body_pose=parameter("smplx_body_pose").reshape(1, -1),
                left_hand_pose=parameter("smplx_left_hand_pose").reshape(1, -1),
                right_hand_pose=parameter("smplx_right_hand_pose").reshape(1, -1),
                jaw_pose=parameter("smplx_jaw_pose").reshape(1, 3),
                betas=parameter("smplx_shape")[: layer.num_betas].reshape(1, -1),
                transl=parameter("smplx_transl").reshape(1, 3),
            )
        joints = body.joints[0].detach().float().cpu().numpy().astype(np.float64)
        vertices = body.vertices[0].detach().float().cpu().numpy().astype(np.float64)
        root_rotation = Rotation.from_rotvec(
            parameter("smplx_root_pose").reshape(3).detach().cpu().numpy()
        ).as_matrix()
        if not are_world:
            joints, vertices = transform_points(camera, joints), transform_points(camera, vertices)
            root_rotation = camera[:3, :3] @ root_rotation
        output.append(
            {
                "root": transform_points(evaluation_gauge, joints[0]),
                "joints": transform_points(evaluation_gauge, joints),
                "vertices": transform_points(evaluation_gauge, vertices),
                "root_rotation": evaluation_gauge[:3, :3] @ root_rotation,
            }
        )
    return output


def human_errors(
    person: dict[str, Any] | None,
    target: dict[str, Any] | None,
    target_head: np.ndarray | None,
    layer: SMPL_Layer,
) -> dict[str, float | bool]:
    if person is None or target is None:
        return {metric: float("nan") for metric in HUMAN_METRICS} | {"available": False}
    predicted_joints = np.asarray(person["joints"], dtype=np.float64)
    target_joints = np.asarray(target["joints"], dtype=np.float64)
    predicted_vertices = np.asarray(person["vertices"], dtype=np.float64)
    target_vertices = np.asarray(target["vertices"], dtype=np.float64)
    joint_count = min(len(predicted_joints), len(target_joints))
    vertex_count = min(len(predicted_vertices), len(target_vertices))
    head_index = int(layer.joint_names.index("head"))
    return {
        "root_error_m": float(np.linalg.norm(np.asarray(person["root"]) - target["root"])),
        "joint_error_m": float(
            np.linalg.norm(predicted_joints[:joint_count] - target_joints[:joint_count], axis=1).mean()
        ),
        "vertex_error_m": float(
            np.linalg.norm(predicted_vertices[:vertex_count] - target_vertices[:vertex_count], axis=1).mean()
        ),
        # Keep the existing cross96 head metric bit-for-bit comparable.  For
        # some sources it uses an authoritative precomputed head annotation
        # rather than a regenerated SMPL-X joint.
        "head_error_m": float(np.linalg.norm(
            predicted_joints[head_index]
            - (target_joints[head_index] if target_head is None else target_head)
        )),
        "available": True,
    }


def camera_errors(camera: np.ndarray, target_camera: np.ndarray) -> dict[str, float | bool]:
    camera, target_camera = homogeneous(camera), homogeneous(target_camera)
    translation = float(np.linalg.norm(camera[:3, 3] - target_camera[:3, 3]))
    rotation = rotation_error_deg(camera, target_camera)
    return {
        "camera_translation_m": translation,
        "camera_rotation_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def method_metrics(
    camera: np.ndarray,
    people: list[dict[str, Any]],
    target_camera: np.ndarray,
    targets: list[dict[str, Any]],
    target_head: np.ndarray | None,
    layer: SMPL_Layer,
) -> dict[str, Any]:
    # This is intentionally only an evaluation pairing under the max_humans=1
    # frozen protocol.  It never enters association, gating, or refinement.
    person = people[0] if len(people) == 1 else None
    target = targets[0] if len(targets) == 1 else None
    return {
        **camera_errors(camera, target_camera),
        **human_errors(person, target, target_head, layer),
        "predicted_person_count": len(people),
        "target_person_count": len(targets),
    }


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / scale) if scale > 1e-10 else float("nan")


def relative_rotvec(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotation vector for the left-multiplicative change target @ reference^-1."""

    return Rotation.from_matrix(np.asarray(target) @ np.asarray(reference).T).as_rotvec()


def shadow_typed_residual(
    shadow_people: list[dict[str, Any]],
    b0_people: list[dict[str, Any]],
    brtc_people: list[dict[str, Any]],
    kabsch_people: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    b0_camera: np.ndarray,
) -> dict[str, Any]:
    """Record, but do not commit, typed shadow-vs-B0 residual evidence.

    This has no authority to change the output.  It is an evaluation-only
    decomposition for Phase 2 of the ICLR blueprint, designed so a later VSP
    policy can be trained/selected from development data without rerunning
    hidden geometry extraction differently.
    """

    if not all(len(group) == 1 for group in (shadow_people, b0_people, brtc_people, kabsch_people, targets)):
        return {"available": False, "reason": "requires_singleton_prediction_and_target"}
    shadow, b0, brtc, kabsch, target = (
        shadow_people[0], b0_people[0], brtc_people[0], kabsch_people[0], targets[0]
    )
    residual = np.asarray(shadow["root"], dtype=np.float64) - np.asarray(b0["root"], dtype=np.float64)
    oracle = np.asarray(target["root"], dtype=np.float64) - np.asarray(b0["root"], dtype=np.float64)
    brtc_shift = np.asarray(brtc["root"], dtype=np.float64) - np.asarray(b0["root"], dtype=np.float64)
    ray = np.asarray(b0["root"], dtype=np.float64) - np.asarray(b0_camera[:3, 3], dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-12)
    shadow_rotation = relative_rotvec(shadow["root_rotation"], b0["root_rotation"])
    oracle_rotation = relative_rotvec(target["root_rotation"], b0["root_rotation"])
    kabsch_rotation = relative_rotvec(kabsch["root_rotation"], b0["root_rotation"])
    return {
        "available": True,
        "root": {
            "shadow_minus_b0_world": residual,
            "oracle_minus_b0_world": oracle,
            "brtc_minus_b0_world": brtc_shift,
            "shadow_norm_m": float(np.linalg.norm(residual)),
            "oracle_norm_m": float(np.linalg.norm(oracle)),
            "brtc_norm_m": float(np.linalg.norm(brtc_shift)),
            "shadow_parallel_ray_m": float(np.dot(residual, ray)),
            "oracle_parallel_ray_m": float(np.dot(oracle, ray)),
            "brtc_parallel_ray_m": float(np.dot(brtc_shift, ray)),
            "shadow_orthogonal_ray_m": float(np.linalg.norm(residual - np.dot(residual, ray) * ray)),
            "oracle_orthogonal_ray_m": float(np.linalg.norm(oracle - np.dot(oracle, ray) * ray)),
            "cosine_shadow_oracle": cosine(residual, oracle),
            "cosine_brtc_oracle": cosine(brtc_shift, oracle),
            "cosine_shadow_brtc": cosine(residual, brtc_shift),
            "shadow_to_oracle_magnitude_ratio": float(
                np.linalg.norm(residual) / max(np.linalg.norm(oracle), 1e-10)
            ),
            "shadow_error_m": float(np.linalg.norm(np.asarray(shadow["root"]) - target["root"])),
            "b0_error_m": float(np.linalg.norm(np.asarray(b0["root"]) - target["root"])),
            "brtc_error_m": float(np.linalg.norm(np.asarray(brtc["root"]) - target["root"])),
        },
        "orientation": {
            "shadow_minus_b0_rotvec_rad": shadow_rotation,
            "oracle_minus_b0_rotvec_rad": oracle_rotation,
            "kabsch_minus_b0_rotvec_rad": kabsch_rotation,
            "shadow_angle_deg": float(np.degrees(np.linalg.norm(shadow_rotation))),
            "oracle_angle_deg": float(np.degrees(np.linalg.norm(oracle_rotation))),
            "kabsch_angle_deg": float(np.degrees(np.linalg.norm(kabsch_rotation))),
            "cosine_shadow_oracle": cosine(shadow_rotation, oracle_rotation),
            "cosine_kabsch_oracle": cosine(kabsch_rotation, oracle_rotation),
            "cosine_shadow_kabsch": cosine(shadow_rotation, kabsch_rotation),
            "shadow_error_deg": float(np.degrees(np.linalg.norm(relative_rotvec(shadow["root_rotation"], target["root_rotation"]))),),
            "b0_error_deg": float(np.degrees(np.linalg.norm(relative_rotvec(b0["root_rotation"], target["root_rotation"]))),),
            "kabsch_error_deg": float(np.degrees(np.linalg.norm(relative_rotvec(kabsch["root_rotation"], target["root_rotation"]))),),
        },
    }


def correlation(first: list[float], second: list[float]) -> float:
    first_array, second_array = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    if int(valid.sum()) < 3 or np.std(first_array[valid]) <= 1e-12 or np.std(second_array[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first_array[valid], second_array[valid])[0, 1])


def typed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row["shadow_typed_residual"] for row in rows if row["shadow_typed_residual"]["available"]]
    result: dict[str, Any] = {"available_cases": len(available), "case_count": len(rows)}
    if not available:
        return result
    root = [row["root"] for row in available]
    orientation = [row["orientation"] for row in available]
    root_keys = (
        "shadow_norm_m", "oracle_norm_m", "brtc_norm_m",
        "shadow_parallel_ray_m", "oracle_parallel_ray_m", "brtc_parallel_ray_m",
        "shadow_orthogonal_ray_m", "oracle_orthogonal_ray_m",
        "cosine_shadow_oracle", "cosine_brtc_oracle", "cosine_shadow_brtc",
        "shadow_to_oracle_magnitude_ratio", "shadow_error_m", "b0_error_m", "brtc_error_m",
    )
    orientation_keys = (
        "shadow_angle_deg", "oracle_angle_deg", "kabsch_angle_deg",
        "cosine_shadow_oracle", "cosine_kabsch_oracle", "cosine_shadow_kabsch",
        "shadow_error_deg", "b0_error_deg", "kabsch_error_deg",
    )
    result["root"] = {key: finite_stats([float(row[key]) for row in root]) for key in root_keys}
    result["orientation"] = {
        key: finite_stats([float(row[key]) for row in orientation]) for key in orientation_keys
    }
    result["root"].update(
        {
            "parallel_signed_correlation_shadow_oracle": correlation(
                [float(row["shadow_parallel_ray_m"]) for row in root],
                [float(row["oracle_parallel_ray_m"]) for row in root],
            ),
            "parallel_signed_correlation_brtc_oracle": correlation(
                [float(row["brtc_parallel_ray_m"]) for row in root],
                [float(row["oracle_parallel_ray_m"]) for row in root],
            ),
            "shadow_direct_root_improvement_rate": float(np.mean([
                row["shadow_error_m"] < row["b0_error_m"] for row in root
            ])),
            "shadow_direct_root_harm_over_5cm_rate": float(np.mean([
                row["shadow_error_m"] - row["b0_error_m"] > 0.05 for row in root
            ])),
        }
    )
    result["orientation"].update(
        {
            "cosine_valid_count": int(sum(np.isfinite(row["cosine_shadow_oracle"]) for row in orientation)),
            "shadow_direct_orientation_improvement_rate": float(np.mean([
                row["shadow_error_deg"] < row["b0_error_deg"] for row in orientation
            ])),
        }
    )
    return result


def evaluate_case(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    gt_views: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    add_mhmr_inputs(gt_views)
    clean = todevice(model_batch_from_gt(gt_views), device)
    shadow_views = set_event_indices(clean, {2})
    raw_views = set_event_indices(clean[2:3], set())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False
        )
        raw, _ = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    pre_camera = homogeneous(camera_matrix(shadow[1]))
    shadow_camera = homogeneous(camera_matrix(shadow[2]))
    raw_camera = homogeneous(camera_matrix(raw[0]))
    boundary = (
        boundary_from_camera_predictions(shadow[2], raw[0])[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    b0_camera = boundary @ raw_camera
    parity = {
        "translation_m": float(np.linalg.norm(shadow_camera[:3, 3] - b0_camera[:3, 3])),
        "matrix_max_abs": float(np.max(np.abs(shadow_camera - b0_camera))),
    }
    if parity["matrix_max_abs"] > 1e-5:
        raise RuntimeError(f"B0 camera parity failed: {parity}")

    gt_pre = homogeneous(gt_pose_from_view(gt_views[1]).detach().float().cpu().numpy())
    gt_post = homogeneous(gt_pose_from_view(gt_views[2]).detach().float().cpu().numpy())
    evaluation_gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = evaluation_gauge @ gt_post
    targets = gt_people_world(gt_views[2], evaluation_gauge, layer)
    target_head = gt_head_world(gt_views[2], evaluation_gauge, layer)

    pre_people = layer_humans(shadow[1], shadow_views[1], {}, layer)
    shadow_people = layer_humans(shadow[2], shadow_views[2], {}, layer)
    raw_people = layer_humans(raw[0], raw_views[0], {}, layer)
    b0_people = [transform_person(boundary, person) for person in raw_people]
    matches = ((0, 0),) if len(pre_people) == 1 and len(b0_people) == 1 else ()
    brtc_people, brtc_debug = refine_matched_people(
        pre_camera, b0_camera, pre_people, b0_people, matches, BRTC_CONFIG
    )
    kabsch_people, kabsch_debug = refine_matched_people_orientation_kabsch(
        pre_camera,
        b0_camera,
        pre_people,
        b0_people,
        matches,
        BRTC_CONFIG,
        DEFAULT_ORIENTATION_KABSCH_CONFIG,
    )
    # Human refinement has no authority to alter the camera: make this a hard
    # per-case invariant rather than a statement in the report.
    for name, camera in (("brtc", b0_camera), ("kabsch", b0_camera)):
        if not np.array_equal(camera, b0_camera):
            raise RuntimeError(f"{name} camera changed from B0")

    return json_ready(
        {
            "timing_seconds": elapsed,
            "b0_camera_parity": parity,
            "singleton_association": {
                "matches": matches,
                "policy": "singleton_only; otherwise exact B0 fallback",
            },
            "diagnostics": {"brtc": brtc_debug, "kabsch": kabsch_debug},
            "shadow_typed_residual": shadow_typed_residual(
                shadow_people, b0_people, brtc_people, kabsch_people, targets, b0_camera
            ),
            "methods": {
                "raw_reset": method_metrics(
                    raw_camera, raw_people, target_camera, targets, target_head, layer
                ),
                "shadow_event_diagnostic": method_metrics(
                    shadow_camera, shadow_people, target_camera, targets, target_head, layer
                ),
                "b0_runtime": method_metrics(
                    b0_camera, b0_people, target_camera, targets, target_head, layer
                ),
                "brtc_lc_v1": method_metrics(
                    b0_camera, brtc_people, target_camera, targets, target_head, layer
                ),
                "brtc_lc_v1_torso4_kabsch_candidate": method_metrics(
                    b0_camera, kabsch_people, target_camera, targets, target_head, layer
                ),
            },
        }
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(rows), "methods": {}}
    for method in METHODS:
        values = [row["methods"][method] for row in rows]
        result["methods"][method] = {
            **{metric: finite_stats([float(row[metric]) for row in values]) for metric in CAMERA_METRICS},
            **{metric: finite_stats([float(row[metric]) for row in values]) for metric in HUMAN_METRICS},
            "human_metric_coverage": float(np.mean([bool(row["available"]) for row in values])) if values else float("nan"),
            "catastrophic_count": int(sum(bool(row["catastrophic"]) for row in values)),
            "catastrophic_rate": float(np.mean([bool(row["catastrophic"]) for row in values])) if values else float("nan"),
        }
        if method != "b0_runtime":
            result["methods"][method]["paired_vs_b0"] = {
                metric: {
                    "mean_delta": float(np.nanmean([
                        row["methods"][method][metric] - row["methods"]["b0_runtime"][metric]
                        for row in rows
                    ])),
                    "improvement_rate": float(np.mean([
                        row["methods"][method][metric] < row["methods"]["b0_runtime"][metric]
                        for row in rows
                        if np.isfinite(row["methods"][method][metric])
                        and np.isfinite(row["methods"]["b0_runtime"][metric])
                    ])) if any(
                        np.isfinite(row["methods"][method][metric])
                        and np.isfinite(row["methods"]["b0_runtime"][metric])
                        for row in rows
                    ) else float("nan"),
                }
                for metric in HUMAN_METRICS
            }
    result["timing_seconds"] = finite_stats([float(row["timing_seconds"]) for row in rows])
    result["brtc_accepted_cases"] = int(
        sum(row["diagnostics"]["brtc"]["accepted_count"] > 0 for row in rows)
    )
    result["kabsch_applied_cases"] = int(
        sum(row["diagnostics"]["kabsch"]["orientation_applied_count"] > 0 for row in rows)
    )
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# cross96 → B0 → BRTC-LC → Kabsch Compatibility Evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']['path']}`",
        f"Checkpoint SHA256: `{report['checkpoint']['sha256']}`",
        f"Records: `{report['records']}`",
        "",
        "This is the same-checkpoint compatibility evaluation required before any camera and human claims may be combined. The `shadow_event_diagnostic` geometry is **not** deployable; its state/humans are discarded. `b0_runtime` is the clean reset trajectory with only a camera-derived B0. BRTC-LC and Kabsch consume copied B0 humans and never modify a camera.",
        "",
        "The frozen cross96 protocol uses `max_humans=1`. Thus BRTC/Kabsch uses a singleton match only when both sides have exactly one detection; this table is not a multi-person/automatic-ID result and has no evaluable layout-vector metric.",
        "",
        "| Split | N | Method | Cam comp. | Cam P95 | Root (m) | Joint (m) | Vertex (m) | Head (m) | Human cov. | Cat. |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("overall", report["summary"])] + [
        (source, report["by_source"][source]) for source in SOURCE_ORDER if source in report["by_source"]
    ]
    for split, summary in groups:
        for method in METHODS:
            row = summary["methods"][method]
            lines.append(
                f"| {split} | {summary['case_count']} | {method} | "
                f"{row['camera_composite']['mean']:.4f} | {row['camera_composite']['p95']:.4f} | "
                f"{row['root_error_m']['mean']:.4f} | {row['joint_error_m']['mean']:.4f} | "
                f"{row['vertex_error_m']['mean']:.4f} | {row['head_error_m']['mean']:.4f} | "
                f"{row['human_metric_coverage']:.1%} | {row['catastrophic_count']} |"
            )
    overall = report["summary"]
    lines.extend(
        [
            "",
            f"BRTC accepted cases: `{overall['brtc_accepted_cases']}/{overall['case_count']}`; "
            f"Kabsch applied cases: `{overall['kabsch_applied_cases']}/{overall['case_count']}`.",
            f"Failures: `{len(report['failures'])}`. Exact camera B0 invariants are asserted per case.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_sha = sha256(args.model_path) if args.sha256 else "not_requested"

    selected: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for record in read_jsonl(args.records):
        source = str(record["source"])
        if source not in args.sources:
            continue
        if args.max_cases_per_source and counts[source] >= args.max_cases_per_source:
            continue
        selected.append(record)
        counts[source] += 1

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    model_flags = configure_model(model)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()

    rows, failures = [], []
    for index, record in enumerate(selected, start=1):
        name = safe_name(record["pattern_id"])
        path = cases_dir / f"{name}.json"
        cached = None
        if path.is_file() and not args.overwrite:
            cached = json.loads(path.read_text(encoding="utf-8"))
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_case(model, layer, load_event_views(record, args), device)
                row.update({"status": "ok", "source": record["source"], "record": record})
            except Exception as error:  # keep a transparent ledger and continue the frozen sweep
                row = {
                    "status": "failed",
                    "source": record["source"],
                    "record": record,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            rows.append(row)
            metrics = row["methods"]["brtc_lc_v1_torso4_kabsch_candidate"]
            print(
                f"[{index:03d}/{len(selected):03d}] {record['source']} {name} "
                f"root={metrics['root_error_m']:.4f} accepted="
                f"{row['diagnostics']['brtc']['accepted_count']} "
                f"kabsch={row['diagnostics']['kabsch']['orientation_applied_count']}",
                flush=True,
            )
        else:
            failures.append(row)
            print(f"[{index:03d}/{len(selected):03d}] {name} FAILED: {row['error']}", flush=True)
        if device.type == "cuda" and index % 5 == 0:
            torch.cuda.empty_cache()

    report = {
        "experiment": "cross96_b0_brtc_lc_v1_torso4_kabsch_compatibility",
        "checkpoint": {"path": str(args.model_path), "sha256": checkpoint_sha},
        "records": str(args.records),
        "source_protocol": "frozen cross96 single-human max_humans=1",
        "model_flags": model_flags,
        "frozen_policies": {
            "brtc_lc_v1": {
                "runtime": "versions/v14/b0_person_triangulation.py",
                "config": json_ready(BRTC_CONFIG.__dict__),
            },
            "torso4_kabsch_candidate": {
                "runtime": "versions/v14/b0_person_triangulation_orientation_kabsch.py",
                "config": json_ready(DEFAULT_ORIENTATION_KABSCH_CONFIG.__dict__),
                "status": "qualified candidate; not the current default runtime",
            },
        },
        "summary": summarize(rows),
        "shadow_typed_summary": typed_summary(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in args.sources
        },
        "shadow_typed_by_source": {
            source: typed_summary([row for row in rows if row["source"] == source])
            for source in args.sources
        },
        "failures": failures,
        "cases": rows,
    }
    report = json_ready(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
