#!/usr/bin/env python3
"""V11 stage-2 gauge-neutral first-write oracle on real camera cuts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import roma
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import dust3r.model as human3r_model_module  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from v10_latent_activation_patching_probe import (  # noqa: E402
    LatentController,
    PatchSpec,
    run_branch,
    source_dict,
)
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    evaluate_case,
    read_jsonl,
    record_spec,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v11_gauge_neutral_first_write" / "stage2_oracle"
DEFAULT_CANDIDATES = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"
KEEP_GPU_OUTPUTS = {
    "camera_pose",
    "pts3d_in_self_view",
    "smpl_rotmat",
    "smpl_transl",
    "smpl_shape",
    "smpl_expression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate_root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--point_sample", type=int, default=4096)
    parser.add_argument("--loss_point_sample", type=int, default=2048)
    parser.add_argument("--opt_steps", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--max_state_residual_std", type=float, default=0.50)
    parser.add_argument("--state_regularization", type=float, default=1e-3)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cpu_predictions(predictions: list[dict]) -> list[dict]:
    output = []
    for prediction in predictions:
        output.append(
            {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in prediction.items()
            }
        )
    return output


@contextmanager
def gpu_output_filter():
    original = human3r_model_module.to_cpu

    def keep_selected(value):
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if key in KEEP_GPU_OUTPUTS}
        return value

    human3r_model_module.to_cpu = keep_selected
    try:
        yield
    finally:
        human3r_model_module.to_cpu = original


def camera_pose_gpu(prediction: dict) -> torch.Tensor:
    return pose_encoding_to_camera(prediction["camera_pose"].float())[0]


def relative_pose_gpu(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(reference) @ current


def rotation_chordal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:3, :3] - b[:3, :3]).square().mean()


def rotation_batch_chordal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    count = min(a.shape[-3], b.shape[-3])
    return (a[..., :count, :, :] - b[..., :count, :, :]).square().mean()


def safe_human(prediction: dict) -> bool:
    value = prediction.get("smpl_transl")
    return isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[1] > 0


def build_loss_targets(
    reset_predictions: list[dict],
    teacher_post: list[dict],
    views: list[dict],
    device: torch.device,
    sample_count: int,
) -> dict:
    gt_poses = [gt_pose_from_view(view).to(device=device, dtype=torch.float32) for view in views]
    teacher_points = [
        prediction["pts3d_in_self_view"].detach().to(device=device, dtype=torch.float32).reshape(-1, 3)
        for prediction in teacher_post
    ]
    reset_points = [
        prediction["pts3d_in_self_view"].detach().to(device=device, dtype=torch.float32).reshape(-1, 3)
        for prediction in reset_predictions
    ]
    point_ids = []
    for offset, (reset_point, teacher_point) in enumerate(zip(reset_points, teacher_points)):
        count = min(len(reset_point), len(teacher_point))
        valid = torch.isfinite(reset_point[:count]).all(dim=-1) & torch.isfinite(teacher_point[:count]).all(dim=-1)
        ids = torch.nonzero(valid, as_tuple=False).flatten()
        if len(ids) > sample_count:
            generator = torch.Generator(device=device)
            generator.manual_seed(9137 + offset)
            order = torch.randperm(len(ids), generator=generator, device=device)[:sample_count]
            ids = ids[order]
        point_ids.append(ids)

    gt_roots = []
    gt_rotmats = []
    for view in views:
        valid = bool(view["smpl_mask"][0, 0].detach().item()) if "smpl_mask" in view else False
        if valid:
            gt_roots.append(view["smpl_j3d"][0, 0, 0].detach().to(device=device, dtype=torch.float32))
            gt_rotmats.append(view["smpl_rotmat"][0, 0].detach().to(device=device, dtype=torch.float32))
        else:
            gt_roots.append(None)
            gt_rotmats.append(None)
    reset_poses = [
        camera_pose_gpu(prediction).detach().to(device=device, dtype=torch.float32)
        for prediction in reset_predictions
    ]
    reset_pose0 = reset_poses[0]
    gt_pose0 = gt_poses[0]
    reset_root0 = reset_predictions[0].get("smpl_transl")
    reset_root0 = reset_root0[0, 0].detach().to(device=device, dtype=torch.float32) if safe_human(reset_predictions[0]) else None
    reset_body0 = reset_predictions[0].get("smpl_rotmat")
    reset_body0 = reset_body0[0, 0].detach().to(device=device, dtype=torch.float32) if isinstance(reset_body0, torch.Tensor) and reset_body0.shape[1] > 0 else None
    scales = []
    for offset in range(len(reset_predictions)):
        reset_rel = relative_pose_gpu(reset_pose0, reset_poses[offset])
        gt_rel = relative_pose_gpu(gt_pose0, gt_poses[offset])
        ids = point_ids[offset]
        if len(ids):
            point_error = torch.linalg.vector_norm(reset_points[offset][ids] - teacher_points[offset][ids], dim=-1).mean()
            depth_error = (reset_points[offset][ids, 2] - teacher_points[offset][ids, 2]).abs().mean()
        else:
            point_error = torch.tensor(0.05, device=device)
            depth_error = torch.tensor(0.05, device=device)
        human_root_error = torch.tensor(0.05, device=device)
        torso_error = torch.tensor(3e-4, device=device)
        local_pose_error = torch.tensor(0.01, device=device)
        if (
            offset > 0
            and reset_root0 is not None
            and reset_body0 is not None
            and gt_roots[0] is not None
            and gt_rotmats[0] is not None
            and safe_human(reset_predictions[offset])
            and gt_roots[offset] is not None
            and gt_rotmats[offset] is not None
        ):
            reset_root = reset_predictions[offset]["smpl_transl"][0, 0].detach().to(device=device, dtype=torch.float32)
            reset_body = reset_predictions[offset]["smpl_rotmat"][0, 0].detach().to(device=device, dtype=torch.float32)
            reset_root_local = reset_rel[:3, :3] @ reset_root + reset_rel[:3, 3]
            gt_root_local = gt_rel[:3, :3] @ gt_roots[offset] + gt_rel[:3, 3]
            human_root_error = torch.linalg.vector_norm(
                (reset_root_local - reset_root0) - (gt_root_local - gt_roots[0])
            )
            reset_torso = reset_rel[:3, :3] @ reset_body[0]
            gt_torso = gt_rel[:3, :3] @ gt_rotmats[offset][0]
            torso_error = (
                (reset_body0[0].transpose(0, 1) @ reset_torso)
                - (gt_rotmats[0][0].transpose(0, 1) @ gt_torso)
            ).square().mean()
            local_pose_error = rotation_batch_chordal(reset_body[1:], gt_rotmats[offset][1:])
        scales.append(
            {
                "camera_translation": torch.linalg.vector_norm(reset_rel[:3, 3] - gt_rel[:3, 3]).clamp_min(0.05),
                "camera_rotation": rotation_chordal(reset_rel, gt_rel).clamp_min(3e-4),
                "pointmap": point_error.clamp_min(0.05),
                "depth": depth_error.clamp_min(0.05),
                "human_relative_root": human_root_error.clamp_min(0.05),
                "human_torso": torso_error.clamp_min(3e-4),
                "human_local_pose": local_pose_error.clamp_min(0.01),
            }
        )
    return {
        "gt_poses": gt_poses,
        "teacher_points": teacher_points,
        "point_ids": point_ids,
        "gt_roots": gt_roots,
        "gt_rotmats": gt_rotmats,
        "baseline_scales": scales,
    }


def gauge_neutral_loss(predictions: list[dict], targets: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    count = min(len(predictions), len(targets["gt_poses"]))
    offsets = [offset for offset in (1, 2, 4, 8) if offset < count]
    if not offsets:
        offsets = list(range(1, count))
    pred_pose0 = camera_pose_gpu(predictions[0]).detach()
    gt_pose0 = targets["gt_poses"][0]
    pred_root0 = predictions[0].get("smpl_transl")
    if safe_human(predictions[0]):
        pred_root0 = pred_root0[0, 0].float().detach()
    else:
        pred_root0 = None
    pred_body0 = predictions[0].get("smpl_rotmat")
    if isinstance(pred_body0, torch.Tensor) and pred_body0.shape[1] > 0:
        pred_body0 = pred_body0[0, 0].float().detach()
    else:
        pred_body0 = None
    gt_root0 = targets["gt_roots"][0]
    gt_body0 = targets["gt_rotmats"][0]

    terms: dict[str, list[torch.Tensor]] = {
        "camera_translation": [],
        "camera_rotation": [],
        "pointmap": [],
        "depth": [],
        "human_relative_root": [],
        "human_torso": [],
        "human_local_pose": [],
    }
    for offset in offsets:
        prediction = predictions[offset]
        pred_pose = camera_pose_gpu(prediction)
        pred_rel = relative_pose_gpu(pred_pose0, pred_pose)
        gt_rel = relative_pose_gpu(gt_pose0, targets["gt_poses"][offset])
        scale = targets["baseline_scales"][offset]
        terms["camera_translation"].append(
            F.smooth_l1_loss(
                (pred_rel[:3, 3] - gt_rel[:3, 3]) / scale["camera_translation"],
                torch.zeros_like(pred_rel[:3, 3]),
            )
        )
        terms["camera_rotation"].append(rotation_chordal(pred_rel, gt_rel) / scale["camera_rotation"])

        points = prediction["pts3d_in_self_view"].float().reshape(-1, 3)
        ids = targets["point_ids"][offset]
        if len(ids):
            target_points = targets["teacher_points"][offset][ids]
            terms["pointmap"].append(
                F.smooth_l1_loss(
                    (points[ids] - target_points) / scale["pointmap"],
                    torch.zeros_like(target_points),
                )
            )
            terms["depth"].append(
                F.smooth_l1_loss(
                    (points[ids, 2] - target_points[:, 2]) / scale["depth"],
                    torch.zeros_like(target_points[:, 2]),
                )
            )

        if (
            pred_root0 is not None
            and pred_body0 is not None
            and gt_root0 is not None
            and gt_body0 is not None
            and safe_human(prediction)
            and targets["gt_roots"][offset] is not None
            and targets["gt_rotmats"][offset] is not None
        ):
            pred_root = prediction["smpl_transl"][0, 0].float()
            pred_body = prediction["smpl_rotmat"][0, 0].float()
            gt_root = targets["gt_roots"][offset]
            gt_body = targets["gt_rotmats"][offset]
            pred_root_local = pred_rel[:3, :3] @ pred_root + pred_rel[:3, 3]
            gt_root_local = gt_rel[:3, :3] @ gt_root + gt_rel[:3, 3]
            terms["human_relative_root"].append(
                F.smooth_l1_loss(
                    ((pred_root_local - pred_root0) - (gt_root_local - gt_root0))
                    / scale["human_relative_root"],
                    torch.zeros_like(pred_root_local),
                )
            )
            pred_torso = pred_rel[:3, :3] @ pred_body[0]
            gt_torso = gt_rel[:3, :3] @ gt_body[0]
            terms["human_torso"].append(
                (
                    (pred_body0[0].transpose(0, 1) @ pred_torso)
                    - (gt_body0[0].transpose(0, 1) @ gt_torso)
                ).square().mean()
                / scale["human_torso"]
            )
            terms["human_local_pose"].append(
                rotation_batch_chordal(pred_body[1:], gt_body[1:]) / scale["human_local_pose"]
            )

    means = {
        key: torch.stack(values).mean() if values else pred_pose0.new_zeros(())
        for key, values in terms.items()
    }
    maxima = {
        key: torch.stack(values).max() if values else pred_pose0.new_zeros(())
        for key, values in terms.items()
    }
    balanced = {key: 0.5 * means[key] + 0.5 * maxima[key] for key in means}
    total = (
        1.0 * balanced["camera_translation"]
        + 1.0 * balanced["camera_rotation"]
        + 0.8 * balanced["pointmap"]
        + 0.2 * balanced["depth"]
        + 0.5 * balanced["human_relative_root"]
        + 0.25 * balanced["human_torso"]
        + 0.10 * balanced["human_local_pose"]
    )
    return total, {
        key: {
            "mean": float(means[key].detach().cpu()),
            "max": float(maxima[key].detach().cpu()),
        }
        for key in means
    }


def differentiable_rollout(
    model,
    views: list[dict],
    device: torch.device,
    corrected_state: torch.Tensor,
) -> list[dict]:
    patch = PatchSpec("gauge_neutral_first_write", ("first_write_state",))
    source = {"new_state": corrected_state}
    with gpu_output_filter(), LatentController(model, 0, False, patch, source, seed=0):
        predictions, _ = model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=False,
        )
    return predictions


def optimize_first_write(
    model,
    views: list[dict],
    base_state_cpu: torch.Tensor,
    targets: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    device = torch.device(args.device)
    base = base_state_cpu.to(device=device, dtype=torch.float32)
    state_scale = base.detach().std(unbiased=False).clamp_min(1e-4)
    raw = torch.nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.Adam([raw], lr=float(args.learning_rate))
    history = []
    best_loss = float("inf")
    best_state = base.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(int(args.opt_steps)):
        optimizer.zero_grad(set_to_none=True)
        delta = state_scale * float(args.max_state_residual_std) * torch.tanh(raw)
        corrected = base + delta
        predictions = differentiable_rollout(model, views, device, corrected)
        data_loss, components = gauge_neutral_loss(predictions, targets, args)
        regularization = float(args.state_regularization) * raw.square().mean()
        loss = data_loss + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        relative_delta = float(torch.linalg.vector_norm(delta.detach()) / torch.linalg.vector_norm(base).clamp_min(1e-8))
        history.append(
            {
                "step": step,
                "loss": value,
                "data_loss": float(data_loss.detach().cpu()),
                "relative_state_delta": relative_delta,
                "components": components,
            }
        )
        if value < best_loss:
            best_loss = value
            best_state = corrected.detach().clone()
        del predictions, corrected, delta, data_loss, regularization, loss
    with torch.no_grad():
        final_delta = state_scale * float(args.max_state_residual_std) * torch.tanh(raw)
        final_state = base + final_delta
        final_predictions = differentiable_rollout(model, views, device, final_state)
        final_data_loss, final_components = gauge_neutral_loss(final_predictions, targets, args)
        final_loss = final_data_loss + float(args.state_regularization) * raw.square().mean()
        final_value = float(final_loss.detach().cpu())
        history.append(
            {
                "step": int(args.opt_steps),
                "loss": final_value,
                "data_loss": float(final_data_loss.detach().cpu()),
                "relative_state_delta": float(
                    torch.linalg.vector_norm(final_delta) / torch.linalg.vector_norm(base).clamp_min(1e-8)
                ),
                "components": final_components,
                "post_update_evaluation": True,
            }
        )
        if final_value < best_loss:
            best_loss = final_value
            best_state = final_state.detach().clone()
        del final_predictions, final_state, final_delta, final_data_loss, final_loss
    return best_state.cpu(), {
        "best_loss": best_loss,
        "history": history,
        "state_scale": float(state_scale.detach().cpu()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def fixed_explicit_transform(root: Path, case_name: str) -> tuple[np.ndarray, str]:
    path = root / case_name / "case_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform_in_human3r_gauge = np.asarray(payload["fixed_explicit"]["transform"], dtype=np.float32)
    gt_to_human3r_gauge = np.asarray(payload["gt_gauge"]["gauge_transform"], dtype=np.float32)
    transform_in_raw_gt_world = np.linalg.inv(gt_to_human3r_gauge) @ transform_in_human3r_gauge
    return transform_in_raw_gt_world.astype(np.float32), str(payload["fixed_explicit_name"])


def max_boundary_difference(reference: dict, current: dict) -> dict:
    output = {}
    for key in ("camera_pose", "pts3d_in_self_view", "smpl_transl", "smpl_rotmat"):
        a = reference.get(key)
        b = current.get(key)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape:
            output[key] = float((a.float() - b.float()).abs().max())
    output["maximum"] = max(output.values(), default=0.0)
    return output


def random_se3(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    axis = generator.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-8)
    angle = generator.uniform(-math.pi, math.pi)
    rotation = roma.rotvec_to_rotmat(torch.from_numpy((axis * angle).astype(np.float32))).numpy()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = generator.uniform(-3.0, 3.0, size=3)
    return transform


def invariance_check(predictions: list[dict], views: list[dict], seed: int) -> dict:
    pred = [pose_encoding_to_camera(row["camera_pose"].float())[0].numpy().astype(np.float32) for row in predictions]
    gt = [gt_pose_from_view(view).cpu().numpy().astype(np.float32) for view in views]
    transform = random_se3(seed)
    camera_delta = []
    for offset in range(1, len(pred)):
        before_pred = np.linalg.inv(pred[0]) @ pred[offset]
        before_gt = np.linalg.inv(gt[0]) @ gt[offset]
        after_pred = np.linalg.inv(transform @ pred[0]) @ (transform @ pred[offset])
        after_gt = np.linalg.inv(transform @ gt[0]) @ (transform @ gt[offset])
        camera_delta.append(float(np.max(np.abs((before_pred - before_gt) - (after_pred - after_gt)))))
    points = predictions[0]["pts3d_in_self_view"][0].float().reshape(-1, 3).numpy()
    ids = np.flatnonzero(np.isfinite(points).all(axis=1))[:1024]
    world = points[ids] @ pred[0][:3, :3].T + pred[0][:3, 3]
    world_h = np.concatenate([world, np.ones((len(world), 1), dtype=np.float32)], axis=1)
    transformed_world = (transform @ world_h.T).T[:, :3]
    transformed_pose = transform @ pred[0]
    recovered = (transformed_world - transformed_pose[:3, 3]) @ transformed_pose[:3, :3]
    return {
        "camera_relative_loss_max_abs_change": max(camera_delta, default=0.0),
        "camera_frame_pointmap_max_abs_change_m": float(np.max(np.abs(recovered - points[ids]))) if len(ids) else 0.0,
        "root_centered_and_local_human": "unchanged by construction because no world-coordinate input enters the loss",
    }


def evaluate_variant(
    spec: dict,
    predictions: list[dict],
    teacher_predictions: list[dict],
    views: list[dict],
    teacher_warmup: int,
    gt_model,
    pred_layer,
    args: argparse.Namespace,
    case_index: int,
    transform: np.ndarray | None,
) -> dict:
    return evaluate_case(
        spec,
        predictions,
        teacher_predictions,
        views,
        teacher_warmup,
        gt_model,
        pred_layer,
        args,
        case_index,
        world_transform=transform,
        update_smpl_gt=False,
    )


def run_case(model, gt_model, pred_layer, spec: dict, args: argparse.Namespace, case_index: int) -> dict:
    device = torch.device(args.device)
    reset_dataset = build_dataset([spec], False, args)
    teacher_dataset = build_dataset([spec], True, args)
    reset_views = next(iter(torch.utils.data.DataLoader(reset_dataset, batch_size=1, num_workers=0)))
    teacher_views = next(iter(torch.utils.data.DataLoader(teacher_dataset, batch_size=1, num_workers=0)))
    reset_views = configure_views(reset_views, device, model.mhmr_img_res)
    teacher_views = configure_views(teacher_views, device, model.mhmr_img_res)
    teacher_warmup = int(spec["warmup_count"])
    with torch.no_grad():
        gt_model.update_smpl_gt(reset_views)
    reset_predictions, reset_latents, reset_seconds, _ = run_branch(
        model, reset_views, device, 0, capture=True, seed=args.seed + case_index
    )
    teacher_predictions, teacher_latents, teacher_seconds, _ = run_branch(
        model, teacher_views, device, teacher_warmup, capture=True, seed=args.seed + case_index
    )
    teacher_post = teacher_predictions[teacher_warmup:]
    targets = build_loss_targets(
        reset_predictions,
        teacher_post,
        reset_views,
        device,
        int(args.loss_point_sample),
    )
    started = time.perf_counter()
    best_state, optimization = optimize_first_write(
        model,
        reset_views,
        reset_latents["new_state"],
        targets,
        args,
    )
    optimization["elapsed_seconds"] = time.perf_counter() - started
    corrected_predictions, _, corrected_seconds, skipped = run_branch(
        model,
        reset_views,
        device,
        0,
        capture=False,
        patch=PatchSpec("gauge_neutral_first_write", ("first_write_state",)),
        source={"new_state": best_state},
        seed=args.seed + case_index,
    )
    absolute_predictions, _, absolute_seconds, absolute_skipped = run_branch(
        model,
        reset_views,
        device,
        0,
        capture=False,
        patch=PatchSpec("absolute_teacher_state", ("first_write_state",)),
        source=source_dict(teacher_latents),
        seed=args.seed + case_index,
    )
    explicit_transform, explicit_name = fixed_explicit_transform(args.candidate_root, spec["record"]["pattern_id"])
    variants = {
        "reset_gt_boundary": evaluate_variant(
            spec, reset_predictions, teacher_predictions, reset_views, teacher_warmup,
            gt_model, pred_layer, args, case_index, None,
        ),
        "absolute_teacher_state_gt_boundary": evaluate_variant(
            spec, absolute_predictions, teacher_predictions, reset_views, teacher_warmup,
            gt_model, pred_layer, args, case_index, None,
        ),
        "gauge_neutral_oracle_gt_boundary": evaluate_variant(
            spec, corrected_predictions, teacher_predictions, reset_views, teacher_warmup,
            gt_model, pred_layer, args, case_index, None,
        ),
        "reset_fixed_explicit": evaluate_variant(
            spec, reset_predictions, teacher_predictions, reset_views, teacher_warmup,
            gt_model, pred_layer, args, case_index, explicit_transform,
        ),
        "gauge_neutral_oracle_fixed_explicit": evaluate_variant(
            spec, corrected_predictions, teacher_predictions, reset_views, teacher_warmup,
            gt_model, pred_layer, args, case_index, explicit_transform,
        ),
    }
    variants["boundary_output_only_gt_boundary"] = copy.deepcopy(variants["reset_gt_boundary"])
    variants["boundary_output_only_gt_boundary"]["causal_control"] = (
        "Only the boundary output is replaceable; no state write changes, so all future frames equal hard reset."
    )
    report = {
        "case_name": spec["record"]["pattern_id"],
        "record": spec["record"],
        "post_frames": spec["post_frames"],
        "optimization": optimization,
        "boundary_lock": {
            "gauge_neutral_vs_reset": max_boundary_difference(reset_predictions[0], corrected_predictions[0]),
            "same_gt_boundary_transform": True,
            "same_fixed_explicit_transform": True,
        },
        "gauge_neutrality": invariance_check(reset_predictions, reset_views, args.seed + case_index),
        "explicit": {
            "name": explicit_name,
            "transform": explicit_transform.tolist(),
            "coordinate_convention": "V10 Human3R frame-0 gauge converted to raw dataset GT world",
        },
        "timing_seconds": {
            "reset": reset_seconds,
            "teacher": teacher_seconds,
            "corrected_eval": corrected_seconds,
            "absolute_eval": absolute_seconds,
        },
        "skipped_replacements": {
            "gauge_neutral": skipped,
            "absolute": absolute_skipped,
        },
        "variants": variants,
    }
    del reset_views, teacher_views, reset_predictions, teacher_predictions
    del corrected_predictions, absolute_predictions, targets, best_state
    torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V11 stage-2 inference and optimization must run on CUDA")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.records)
    selected = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: int(args.max_cases)]
    specs = [record_spec(record, args) for record in selected]
    output = args.output_dir / f"stage2_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    progress = args.output_dir / f"stage2_progress_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    completed: dict[str, dict] = {}
    if progress.exists() and not args.overwrite:
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_name"]] = row
    if args.overwrite:
        progress.unlink(missing_ok=True)
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, torch.device(args.device))
    started = time.perf_counter()
    cases = []
    for index, spec in enumerate(specs):
        name = spec["record"]["pattern_id"]
        if name in completed:
            cases.append(completed[name])
            continue
        case = run_case(model, gt_model, pred_layer, spec, args, index)
        cases.append(case)
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(case, ensure_ascii=False, allow_nan=True) + "\n")
        before = case["variants"]["reset_gt_boundary"]["mean_future"]
        after = case["variants"]["gauge_neutral_oracle_gt_boundary"]["mean_future"]
        print(
            f">> [{index + 1}/{len(specs)}] {name} "
            f"RPE {before['camera_relative_rotation_deg']:.3f}->{after['camera_relative_rotation_deg']:.3f}deg "
            f"PM {before['camera_frame_pointmap_m']:.3f}->{after['camera_frame_pointmap_m']:.3f}m",
            flush=True,
        )
    report = {
        "experiment": "V11 Stage-2 Gauge-Neutral First-Write Oracle",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "human3r_frozen": True,
            "optimization_device": args.device,
            "boundary_output_locked": True,
            "absolute_world_losses": False,
            "teacher_state_latent_mse": False,
            "explicit_se3_applied_once_for_evaluation": True,
        },
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
