#!/usr/bin/env python3
"""Run the causal V14.2 shadow-Boundary segment probe on one AvatarReX cut."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "config/manifests/v14_2_segment/single/lbn1_1192.jsonl"
)
DEFAULT_OUTPUT = Path("/dev/shm/movie3r_v14_2/lbn1_1192")

METHOD_ORDER = (
    "continue",
    "raw_reset",
    "shadow_fixed_b0",
    "shadow_b0_v16_rotation",
    "shadow_b0_v12_lite",
    "gt_camera_only_boundary",
    "gt_rotation_human_anchor_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise ValueError(f"Expected exactly one manifest record, found {len(records)}")
    record = records[0]
    lengths = {len(record[key]) for key in ("seqs", "frames", "shot_labels")}
    if len(lengths) != 1:
        raise ValueError("Manifest seqs/frames/shot_labels lengths differ")
    event_indices = [i for i, value in enumerate(record["shot_labels"]) if int(value) == 1]
    if len(event_indices) != 1 or event_indices[0] == 0:
        raise ValueError(f"Expected one non-initial cut event, found {event_indices}")
    return record


def raw_calibration_roots() -> dict[str, str]:
    root = Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta")
    return {name: str(root / name) for name in ("lbn1", "lbn2", "zzr", "zxc")}


def load_batch(args: argparse.Namespace, record: dict) -> list[dict]:
    from dust3r.datasets.avatarrex import AvatarReX_Pattern
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.image import pad_image

    dataset = AvatarReX_Pattern(
        allow_repeat=True,
        split="Training",
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=args.size,
        resize_mode="human3r_demo",
        num_views=len(record["frames"]),
        seed=14201,
        n_corres=0,
        manifest_path=str(args.manifest_path.resolve()),
        load_da3_depth=False,
        raw_calibration_root=raw_calibration_roots(),
        max_humans=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    images = torch.stack([view["img"] for view in batch], dim=0)
    images = images.view(-1, *images.shape[2:])
    intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=0)
    intrinsics = intrinsics.view(-1, *intrinsics.shape[2:])
    model_resolution = 896
    intrinsics_mhmr = resize_camera_intrinsics(
        intrinsics, *images.shape[2:], model_resolution
    )
    images_mhmr = pad_image(images, model_resolution)
    for view, image_mhmr, intrinsic_mhmr in zip(
        batch,
        images_mhmr.chunk(len(batch), dim=0),
        intrinsics_mhmr.chunk(len(batch), dim=0),
    ):
        view["img_mhmr"] = image_mhmr
        view["K_mhmr"] = intrinsic_mhmr
    return batch


def model_batch_from_gt(batch: list[dict]) -> list[dict]:
    from dust3r.inference import _make_v8_image_only_model_batch

    model_batch = _make_v8_image_only_model_batch(batch)
    for view in model_batch:
        reference = view["img_mask"]
        view["reset"] = torch.zeros_like(reference, dtype=torch.bool)
        for key in ("update", "update_state", "update_mem", "update_v8_history"):
            view[key] = torch.ones_like(reference, dtype=torch.bool)
    return model_batch


def set_event_indices(views: list[dict], event_indices: set[int]) -> list[dict]:
    routed = copy.deepcopy(views)
    for index, view in enumerate(routed):
        reference = view["img"]
        view["shot_label"] = torch.full(
            (reference.shape[0],),
            1.0 if index in event_indices else 0.0,
            dtype=reference.dtype,
            device=reference.device,
        )
    return routed


def configure_model(model) -> dict:
    model.v9_oracle_correction_gate_enabled = True
    model.v9_oracle_correction_inference_only = False
    model.v9_oracle_correction_cache_enabled = False
    model.v14_1_event_only_head_lora = True
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    keys = (
        "enable_v8_pose_prompt",
        "enable_v8_human_latent_corr",
        "enable_v8_human_trans_corr",
        "enable_v8_head_lora",
        "v9_oracle_correction_gate_enabled",
        "v9_oracle_correction_cache_enabled",
        "v14_1_event_only_head_lora",
    )
    return {key: bool(getattr(model, key, False)) for key in keys}


def run_rollout(model, views: list[dict], device: str, name: str) -> tuple[list[dict], list[dict], float]:
    if not views:
        raise ValueError(f"Rollout {name} has no views")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, returned_views = model.forward_recurrent_lighter(
            views, device, ret_state=False, use_ttt3r=False
        )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    print(f">> {name}: {len(views)} frames in {elapsed:.2f}s", flush=True)
    return predictions, returned_views, elapsed


def camera_matrix(prediction: dict) -> np.ndarray:
    from dust3r.utils.camera import pose_encoding_to_camera

    return (
        pose_encoding_to_camera(prediction["camera_pose"].detach().float())[0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def gt_camera_matrix(view: dict) -> np.ndarray:
    value = view.get("raw_camera_pose", view["camera_pose"])
    return value.detach().float()[0].cpu().numpy().astype(np.float32)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ point + transform[:3, 3]).astype(np.float32)


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = estimated[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def boundary_error(estimated: np.ndarray, target: np.ndarray) -> dict:
    return {
        "translation_m": float(np.linalg.norm(estimated[:3, 3] - target[:3, 3])),
        "rotation_deg": rotation_error_deg(estimated, target),
    }


def boundary_with_rotation(boundary: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    output = np.asarray(boundary, dtype=np.float32).copy()
    output[:3, :3] = np.asarray(rotation, dtype=np.float32)
    return output


def root_anchored_boundary(
    rotation: np.ndarray,
    raw_camera: np.ndarray,
    raw_root_camera: np.ndarray,
    target_root_world: np.ndarray,
) -> np.ndarray:
    """Build the one-shot V12 translation after rotation has been selected."""
    raw_root_world = transform_point(raw_camera, raw_root_camera)
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = np.asarray(rotation, dtype=np.float32)
    output[:3, 3] = np.asarray(target_root_world, dtype=np.float32) - (
        output[:3, :3] @ raw_root_world
    )
    return output


def gt_head_world(
    view: dict,
    evaluation_alignment: np.ndarray,
    smpl_layer,
) -> np.ndarray | None:
    mask = view.get("smpl_mask")
    if mask is None:
        return None
    valid = torch.nonzero(mask[0].detach().cpu().bool(), as_tuple=False).flatten()
    if valid.numel() == 0:
        return None
    human_index = int(valid[0])

    precomputed = view.get("smplx_has_precomputed_keypoints")
    has_precomputed = bool(
        precomputed is not None
        and float(precomputed[0, human_index].detach().cpu()) > 0.5
    )
    if has_precomputed:
        head = view.get("smplx_head_world")
        if head is None:
            return None
        point = head[0, human_index].detach().float().cpu().numpy()
    else:
        required = (
            "smplx_root_pose",
            "smplx_body_pose",
            "smplx_left_hand_pose",
            "smplx_right_hand_pose",
            "smplx_jaw_pose",
            "smplx_shape",
            "smplx_transl",
        )
        if any(key not in view for key in required):
            return None
        device = next(smpl_layer.parameters()).device

        def parameter(key: str) -> torch.Tensor:
            return view[key][0, human_index].detach().float().to(device)

        with torch.no_grad():
            output = smpl_layer.bm_x(
                global_orient=parameter("smplx_root_pose").reshape(1, 3),
                body_pose=parameter("smplx_body_pose").reshape(1, -1),
                left_hand_pose=parameter("smplx_left_hand_pose").reshape(1, -1),
                right_hand_pose=parameter("smplx_right_hand_pose").reshape(1, -1),
                jaw_pose=parameter("smplx_jaw_pose").reshape(1, 3),
                betas=parameter("smplx_shape")[: smpl_layer.num_betas].reshape(1, -1),
                transl=parameter("smplx_transl").reshape(1, 3),
            )
        head_index = smpl_layer.joint_names.index("head")
        point = output.joints[0, head_index].detach().float().cpu().numpy()

    params_are_world = view.get("human_params_are_world")
    if params_are_world is not None and not bool(params_are_world[0].detach().cpu()):
        point = transform_point(gt_camera_matrix(view), point)
    if not np.isfinite(point).all():
        return None
    return transform_point(evaluation_alignment, point)


def predicted_head_world(prediction: dict, camera: np.ndarray) -> np.ndarray | None:
    translation = prediction.get("smpl_transl")
    if translation is None or translation.shape[1] == 0:
        return None
    point = translation[0, 0].detach().float().cpu().numpy()
    if not np.isfinite(point).all():
        return None
    return transform_point(camera, point)


def finite_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def evaluate_variant(
    name: str,
    predictions: list[dict],
    target_cameras: list[np.ndarray],
    target_heads: list[np.ndarray | None],
    cut_idx: int,
) -> dict:
    rows = []
    for index, (prediction, target_camera, target_head) in enumerate(
        zip(predictions, target_cameras, target_heads)
    ):
        camera = camera_matrix(prediction)
        translation = float(np.linalg.norm(camera[:3, 3] - target_camera[:3, 3]))
        rotation = rotation_error_deg(camera, target_camera)
        predicted_head = predicted_head_world(prediction, camera)
        human = (
            float(np.linalg.norm(predicted_head - target_head))
            if predicted_head is not None and target_head is not None
            else float("nan")
        )
        rows.append(
            {
                "index": index,
                "translation_m": translation,
                "rotation_deg": rotation,
                "composite": translation + 0.02 * rotation,
                "human_head_m": human,
            }
        )
    post = rows[cut_idx:]
    return {
        "name": name,
        "first_post": post[0],
        "post": {
            key: finite_summary([row[key] for row in post])
            for key in ("translation_m", "rotation_deg", "composite", "human_head_m")
        },
        "catastrophic_rate": float(
            np.mean(
                [row["translation_m"] > 1.0 or row["rotation_deg"] > 30.0 for row in post]
            )
        ),
        "per_frame": rows,
    }


def transformed_predictions(predictions: list[dict], boundary: torch.Tensor) -> list[dict]:
    from dust3r.v14_outputs import apply_boundary_to_prediction

    return [apply_boundary_to_prediction(prediction, boundary) for prediction in predictions]


def merged_predictions(pre: list[dict], post: list[dict]) -> list[dict]:
    return [copy.deepcopy(prediction) for prediction in pre] + post


def merged_views(pre: list[dict], post: list[dict]) -> list[dict]:
    views = [copy.deepcopy(view) for view in pre + post]
    for view in views:
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    return views


def save_viewer_payload(name: str, predictions: list[dict], views: list[dict], output_dir: Path) -> Path:
    from demo import prepare_output

    destination = output_dir / name
    prepare_output(
        {"pred": predictions, "views": views},
        str(destination),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=896,
        subsample=1,
    )
    return destination


def markdown_report(report: dict) -> str:
    lines = [
        "# V14.2 Single-Sequence Shadow Boundary Probe",
        "",
        "The shadow branch sees only pre-cut frames and the first post-cut frame. Its state is discarded. "
        "The raw-reset branch is the only post-cut state, and one fixed `B0` is applied to its full segment.",
        "",
        "| Method | Trans mean (m) | Rot mean (deg) | Composite mean | Human head mean (m) | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_ORDER:
        method = report["methods"][name]
        post = method["post"]
        lines.append(
            f"| {name} | {post['translation_m']['mean']:.4f} | "
            f"{post['rotation_deg']['mean']:.3f} | {post['composite']['mean']:.4f} | "
            f"{post['human_head_m']['mean']:.4f} | {method['catastrophic_rate']:.1%} |"
        )
    audit = report["causal_audit"]
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- Predicted `B0` error: {audit['predicted_b0_error']['translation_m']:.4f} m, "
            f"{audit['predicted_b0_error']['rotation_deg']:.3f} deg.",
            f"- Raw first-only/full consistency: {audit['raw_first_only_vs_full']['translation_m']:.8f} m, "
            f"{audit['raw_first_only_vs_full']['rotation_deg']:.8f} deg.",
            f"- Shadow first-post camera error: {audit['shadow_first_post_error']['translation_m']:.4f} m, "
            f"{audit['shadow_first_post_error']['rotation_deg']:.3f} deg.",
            f"- `B0 + V16` error: {audit['v16_boundary_error']['translation_m']:.4f} m, "
            f"{audit['v16_boundary_error']['rotation_deg']:.3f} deg.",
            f"- `B0 + V12 Lite` error: {audit['v12_boundary_error']['translation_m']:.4f} m, "
            f"{audit['v12_boundary_error']['rotation_deg']:.3f} deg.",
            f"- GT-rotation + human-anchor camera error: "
            f"{audit['gt_human_anchor_camera_boundary_error']['translation_m']:.4f} m, "
            f"{audit['gt_human_anchor_camera_boundary_error']['rotation_deg']:.3f} deg.",
            "- Composite is `translation_m + 0.02 * rotation_deg`.",
            "- Catastrophic means translation > 1 m or rotation > 30 deg.",
            "",
            "`gt_camera_only_boundary` is exact only for the selected GT camera gauge; it is "
            "not a full pointmap or human upper bound. `gt_rotation_human_anchor_oracle` "
            "separately diagnoses visual continuity when local Human3R depth is biased.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record = read_manifest(args.manifest_path)
    cut_idx = record["shot_labels"].index(1)
    gt_batch = load_batch(args, record)
    model_batch = model_batch_from_gt(gt_batch)

    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.smpl_layer import SMPL_Layer
    from dust3r.v14_outputs import boundary_from_camera_predictions
    from scripts.boundary_human3r_reset_support import predicted_human
    from versions.v12.experiments.v14_5_true_recurrent_multicut_audit import (
        conditional_rotation,
    )

    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    model_flags = configure_model(model)
    pred_layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    print(f">> model flags: {model_flags}", flush=True)

    continue_views = set_event_indices(model_batch, set())
    shadow_views = set_event_indices(model_batch[: cut_idx + 1], {cut_idx})
    raw_first_views = set_event_indices(model_batch[cut_idx : cut_idx + 1], set())
    raw_full_views = set_event_indices(model_batch[cut_idx:], set())

    continue_predictions, continue_returned, continue_time = run_rollout(
        model, continue_views, str(device), "continue_event_off"
    )
    shadow_predictions, _, shadow_time = run_rollout(
        model, shadow_views, str(device), "shadow_pre_plus_first_post"
    )
    raw_first_predictions, _, raw_first_time = run_rollout(
        model, raw_first_views, str(device), "raw_reset_first_only"
    )
    raw_predictions, raw_returned, raw_time = run_rollout(
        model, raw_full_views, str(device), "raw_reset_full_post"
    )

    predicted_boundary = boundary_from_camera_predictions(
        shadow_predictions[-1], raw_first_predictions[0]
    )
    predicted_boundary_np = predicted_boundary[0].numpy().astype(np.float32)
    raw_first_only_camera = camera_matrix(raw_first_predictions[0])
    raw_first_full_camera = camera_matrix(raw_predictions[0])

    gt_cameras = [gt_camera_matrix(view) for view in gt_batch]
    evaluation_alignment = camera_matrix(continue_predictions[0]) @ np.linalg.inv(gt_cameras[0])
    target_cameras = [(evaluation_alignment @ pose).astype(np.float32) for pose in gt_cameras]
    target_heads = [
        gt_head_world(view, evaluation_alignment, pred_layer) for view in gt_batch
    ]
    # This is an oracle for the camera pose only. The raw-reset Human3R
    # pointmap and local human can still contain depth, scale, and tilt error,
    # so this must not be presented as a full 3D visualization upper bound.
    gt_camera_boundary_np = target_cameras[cut_idx] @ np.linalg.inv(raw_first_only_camera)

    history_start = max(0, cut_idx - 5)
    v16_rotation, v16_diagnostics = conditional_rotation(
        predicted_boundary_np,
        continue_predictions[history_start:cut_idx],
        gt_batch[history_start:cut_idx],
        raw_predictions[0],
        gt_batch[cut_idx],
        pred_layer,
        None,
        SimpleNamespace(enable_vggt=False),
    )
    v16_boundary_np = boundary_with_rotation(predicted_boundary_np, v16_rotation)

    old_human = predicted_human(
        continue_predictions[cut_idx - 1],
        gt_batch[cut_idx - 1]["camera_intrinsics"],
        pred_layer,
    )
    new_human = predicted_human(
        raw_predictions[0],
        gt_batch[cut_idx]["camera_intrinsics"],
        pred_layer,
    )
    if old_human is not None and new_human is not None:
        old_root_world = transform_point(
            camera_matrix(continue_predictions[cut_idx - 1]), old_human["root"]
        )
        v12_boundary_np = root_anchored_boundary(
            v16_rotation,
            raw_first_only_camera,
            new_human["root"],
            old_root_world,
        )
        v12_translation_branch = "explicit_human_root"
        gt_human_anchor_boundary_np = root_anchored_boundary(
            gt_camera_boundary_np[:3, :3],
            raw_first_only_camera,
            new_human["root"],
            old_root_world,
        )
        gt_human_anchor_branch = "gt_rotation_with_predicted_human_root_anchor"
    else:
        v12_boundary_np = v16_boundary_np.copy()
        v12_translation_branch = "b0_translation_no_valid_human"
        gt_human_anchor_boundary_np = gt_camera_boundary_np.copy()
        gt_human_anchor_branch = "gt_camera_only_no_valid_human"

    v16_boundary = torch.from_numpy(v16_boundary_np).unsqueeze(0)
    v12_boundary = torch.from_numpy(v12_boundary_np).unsqueeze(0)
    gt_camera_boundary = torch.from_numpy(gt_camera_boundary_np).unsqueeze(0)
    gt_human_anchor_boundary = torch.from_numpy(
        gt_human_anchor_boundary_np
    ).unsqueeze(0)

    fixed_post = transformed_predictions(raw_predictions, predicted_boundary)
    v16_post = transformed_predictions(raw_predictions, v16_boundary)
    v12_post = transformed_predictions(raw_predictions, v12_boundary)
    gt_camera_post = transformed_predictions(raw_predictions, gt_camera_boundary)
    gt_human_anchor_post = transformed_predictions(
        raw_predictions, gt_human_anchor_boundary
    )
    raw_merged = merged_predictions(continue_predictions[:cut_idx], raw_predictions)
    fixed_merged = merged_predictions(continue_predictions[:cut_idx], fixed_post)
    v16_merged = merged_predictions(continue_predictions[:cut_idx], v16_post)
    v12_merged = merged_predictions(continue_predictions[:cut_idx], v12_post)
    gt_camera_merged = merged_predictions(continue_predictions[:cut_idx], gt_camera_post)
    gt_human_anchor_merged = merged_predictions(
        continue_predictions[:cut_idx], gt_human_anchor_post
    )
    full_views = merged_views(continue_returned[:cut_idx], raw_returned)

    methods = {
        "continue": evaluate_variant(
            "continue", continue_predictions, target_cameras, target_heads, cut_idx
        ),
        "raw_reset": evaluate_variant(
            "raw_reset", raw_merged, target_cameras, target_heads, cut_idx
        ),
        "shadow_fixed_b0": evaluate_variant(
            "shadow_fixed_b0", fixed_merged, target_cameras, target_heads, cut_idx
        ),
        "shadow_b0_v16_rotation": evaluate_variant(
            "shadow_b0_v16_rotation", v16_merged, target_cameras, target_heads, cut_idx
        ),
        "shadow_b0_v12_lite": evaluate_variant(
            "shadow_b0_v12_lite", v12_merged, target_cameras, target_heads, cut_idx
        ),
        "gt_camera_only_boundary": evaluate_variant(
            "gt_camera_only_boundary",
            gt_camera_merged,
            target_cameras,
            target_heads,
            cut_idx,
        ),
        "gt_rotation_human_anchor_oracle": evaluate_variant(
            "gt_rotation_human_anchor_oracle",
            gt_human_anchor_merged,
            target_cameras,
            target_heads,
            cut_idx,
        ),
    }

    shadow_camera = camera_matrix(shadow_predictions[-1])
    report = {
        "experiment": "V14.2 causal one-shot shadow Boundary segment validation",
        "model_path": str(args.model_path.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "device": str(device),
        "record": record,
        "cut_idx": cut_idx,
        "pre_frame_count": cut_idx,
        "post_frame_count": len(record["frames"]) - cut_idx,
        "model_flags": model_flags,
        "constraints": {
            "causal": True,
            "future_frames_used_for_b0": False,
            "shadow_state_committed": False,
            "raw_reset_state_is_only_committed_state": True,
            "fixed_boundary_for_entire_post_segment": True,
            "local_smpl_parameters_modified_by_boundary": False,
        },
        "timing_seconds": {
            "continue": continue_time,
            "shadow": shadow_time,
            "raw_first_only": raw_first_time,
            "raw_full": raw_time,
        },
        "boundaries": {
            "predicted_b0": predicted_boundary_np.tolist(),
            "b0_v16_rotation": v16_boundary_np.tolist(),
            "b0_v12_lite": v12_boundary_np.tolist(),
            "gt_camera_only_evaluation": gt_camera_boundary_np.tolist(),
            "gt_rotation_human_anchor_diagnostic": (
                gt_human_anchor_boundary_np.tolist()
            ),
        },
        "gt_boundary_semantics": {
            "gt_camera_only_boundary": (
                "Uses GT camera pose to make the first post-cut camera exact. It does "
                "not use GT depth, pointmap, or human geometry and is not a full 3D "
                "visualization upper bound."
            ),
            "gt_rotation_human_anchor_oracle": (
                "Uses GT relative camera rotation, then translates the predicted post-cut "
                "human root onto the last predicted pre-cut root. This diagnoses visual "
                "continuity under local Human3R depth bias, but is not a camera-pose oracle."
            ),
        },
        "postprocess": {
            "v16": v16_diagnostics,
            "v12_translation_branch": v12_translation_branch,
            "gt_human_anchor_branch": gt_human_anchor_branch,
            "v13": (
                "single-human fallback; uniform multi-human consensus is identical to V12 "
                "for this one-person AvatarReX case"
            ),
        },
        "causal_audit": {
            "predicted_b0_error": boundary_error(
                predicted_boundary_np, gt_camera_boundary_np
            ),
            "raw_first_only_vs_full": boundary_error(raw_first_only_camera, raw_first_full_camera),
            "shadow_first_post_error": boundary_error(shadow_camera, target_cameras[cut_idx]),
            "v16_boundary_error": boundary_error(
                v16_boundary_np, gt_camera_boundary_np
            ),
            "v12_boundary_error": boundary_error(
                v12_boundary_np, gt_camera_boundary_np
            ),
            "gt_human_anchor_camera_boundary_error": boundary_error(
                gt_human_anchor_boundary_np, gt_camera_boundary_np
            ),
        },
        "methods": methods,
    }

    output_paths = {
        "continue": save_viewer_payload(
            "continue", continue_predictions, copy.deepcopy(continue_returned), args.output_dir
        ),
        "raw_reset": save_viewer_payload(
            "raw_reset", raw_merged, copy.deepcopy(full_views), args.output_dir
        ),
        "shadow_fixed_b0": save_viewer_payload(
            "shadow_fixed_b0", fixed_merged, copy.deepcopy(full_views), args.output_dir
        ),
        "shadow_b0_v16_rotation": save_viewer_payload(
            "shadow_b0_v16_rotation", v16_merged, copy.deepcopy(full_views), args.output_dir
        ),
        "shadow_b0_v12_lite": save_viewer_payload(
            "shadow_b0_v12_lite", v12_merged, copy.deepcopy(full_views), args.output_dir
        ),
        "gt_camera_only_boundary": save_viewer_payload(
            "gt_camera_only_boundary",
            gt_camera_merged,
            copy.deepcopy(full_views),
            args.output_dir,
        ),
        "gt_rotation_human_anchor_oracle": save_viewer_payload(
            "gt_rotation_human_anchor_oracle",
            gt_human_anchor_merged,
            copy.deepcopy(full_views),
            args.output_dir,
        ),
    }
    report["viewer_outputs"] = {key: str(path) for key, path in output_paths.items()}
    report_path = args.output_dir / "v14_2_single_sequence_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output_dir / "v14_2_single_sequence_report.md"
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
