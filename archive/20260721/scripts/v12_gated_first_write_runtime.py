"""Runtime helpers for V12 first-write prompt training and evaluation."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from types import MethodType

import torch
import torch.nn.functional as F


class GatedFirstWriteController(AbstractContextManager):
    def __init__(
        self,
        human3r,
        adapter,
        pair: dict,
        variant: str,
        source_mode: str = "correct",
        donor_pair: dict | None = None,
        seed: int = 0,
    ) -> None:
        self.human3r = human3r
        self.adapter = adapter
        self.pair = pair
        self.variant = variant
        self.source_mode = source_mode
        self.donor_pair = donor_pair
        self.seed = int(seed)
        self.original_rollout = None
        self.rollout_index = -1
        self.output = None

    def _history(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.donor_pair if self.source_mode == "wrong" and self.donor_pair is not None else self.pair
        state = source["old_state"].to(device=device, dtype=dtype).unsqueeze(0)
        memory = source["old_pose_memory"].to(device=device, dtype=dtype).unsqueeze(0)
        if self.variant == "no_old" or self.source_mode == "zero":
            return torch.zeros_like(state), torch.zeros_like(memory)
        if self.source_mode == "shuffle":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            state_order = torch.randperm(state.shape[1], generator=generator).to(device)
            memory_order = torch.randperm(memory.shape[1], generator=generator).to(device)
            return state[:, state_order], memory[:, memory_order]
        return state, memory

    def __enter__(self):
        self.original_rollout = self.human3r._recurrent_rollout

        def rollout_wrapper(_model, *args, **kwargs):
            self.rollout_index += 1
            result = self.original_rollout(*args, **kwargs)
            if self.rollout_index != 0:
                return result
            old_state, old_memory = self._history(result[0].device, result[0].dtype)
            image_tokens = args[2]
            human_tokens = args[6]
            image_summary = torch.cat(
                [image_tokens.mean(dim=1), image_tokens.std(dim=1, unbiased=False)], dim=-1
            ).float()
            if human_tokens is None or human_tokens.numel() == 0:
                human_summary = image_summary.new_zeros(image_summary.shape[0], 1536)
            else:
                human_summary = torch.cat(
                    [human_tokens.mean(dim=1), human_tokens.std(dim=1, unbiased=False)], dim=-1
                ).float()
            memory_summary = torch.cat(
                [old_memory.mean(dim=1), old_memory.std(dim=1, unbiased=False)], dim=-1
            ).float()
            diagnostics = self.pair["diagnostics"].to(result[0].device).float().reshape(1, -1)
            self.output = self.adapter(
                old_state.float(),
                result[0].float(),
                image_summary,
                human_summary,
                args[4].reshape(args[4].shape[0], -1).float(),
                memory_summary,
                diagnostics,
                gate_override=1.0 if self.variant == "ungated" else None,
            )
            return self.output.corrected_state.to(result[0].dtype), result[1], result[2]

        self.human3r._recurrent_rollout = MethodType(rollout_wrapper, self.human3r)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.human3r._recurrent_rollout = self.original_rollout
        return False


def camera_pose(prediction: dict) -> torch.Tensor:
    from dust3r.utils.camera import pose_encoding_to_camera

    return pose_encoding_to_camera(prediction["camera_pose"].float())[0]


def rotation_chordal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:3, :3] - b[:3, :3]).square().mean()


def cached_targets(pair: dict, device: torch.device) -> dict:
    target = pair["loss_targets"]
    return {
        "gt_poses": target["gt_poses"].to(device).float(),
        "point_ids": [value.to(device).long() for value in target["point_ids"]],
        "teacher_point_samples": [value.to(device).float() for value in target["teacher_point_samples"]],
        "gt_roots": target["gt_roots"].to(device).float(),
        "gt_rotmats": target["gt_rotmats"].to(device).float(),
        "human_valid": target["human_valid"].to(device),
        "baseline_scales": {key: value.to(device).float() for key, value in target["baseline_scales"].items()},
    }


def cached_gauge_neutral_loss(predictions: list[dict], pair: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    target = cached_targets(pair, device)
    count = min(len(predictions), len(target["gt_poses"]))
    offsets = [offset for offset in (1, 2, 4, 8) if offset < count]
    pred_pose0 = camera_pose(predictions[0]).detach()
    gt_pose0 = target["gt_poses"][0]
    pred_root0 = predictions[0].get("smpl_transl")
    pred_root0 = pred_root0[0, 0].float().detach() if pred_root0 is not None and pred_root0.shape[1] else None
    pred_body0 = predictions[0].get("smpl_rotmat")
    pred_body0 = pred_body0[0, 0].float().detach() if pred_body0 is not None and pred_body0.shape[1] else None
    terms = {key: [] for key in ("camera_t", "camera_r", "pointmap", "depth", "human_root", "torso")}
    for offset in offsets:
        pose = camera_pose(predictions[offset])
        pred_rel = torch.linalg.inv(pred_pose0) @ pose
        gt_rel = torch.linalg.inv(gt_pose0) @ target["gt_poses"][offset]
        scales = {key: value[offset].clamp_min(1e-8) for key, value in target["baseline_scales"].items()}
        terms["camera_t"].append(
            F.smooth_l1_loss(
                (pred_rel[:3, 3] - gt_rel[:3, 3]) / scales["camera_translation"],
                torch.zeros(3, device=device),
            )
        )
        terms["camera_r"].append(rotation_chordal(pred_rel, gt_rel) / scales["camera_rotation"])
        points = predictions[offset]["pts3d_in_self_view"].float().reshape(-1, 3)
        ids = target["point_ids"][offset]
        sample = target["teacher_point_samples"][offset]
        if len(ids):
            terms["pointmap"].append(
                F.smooth_l1_loss(
                    (points[ids] - sample) / scales["pointmap"],
                    torch.zeros_like(sample),
                )
            )
            terms["depth"].append(
                F.smooth_l1_loss(
                    (points[ids, 2] - sample[:, 2]) / scales["depth"],
                    torch.zeros_like(sample[:, 2]),
                )
            )
        prediction = predictions[offset]
        if (
            pred_root0 is not None
            and pred_body0 is not None
            and bool(target["human_valid"][0])
            and bool(target["human_valid"][offset])
            and prediction.get("smpl_transl") is not None
            and prediction["smpl_transl"].shape[1] > 0
        ):
            root = prediction["smpl_transl"][0, 0].float()
            body = prediction["smpl_rotmat"][0, 0].float()
            pred_root_local = pred_rel[:3, :3] @ root + pred_rel[:3, 3]
            gt_root_local = gt_rel[:3, :3] @ target["gt_roots"][offset] + gt_rel[:3, 3]
            terms["human_root"].append(
                F.smooth_l1_loss(
                    ((pred_root_local - pred_root0) - (gt_root_local - target["gt_roots"][0]))
                    / scales["human_relative_root"],
                    torch.zeros(3, device=device),
                )
            )
            pred_torso = pred_rel[:3, :3] @ body[0]
            gt_torso = gt_rel[:3, :3] @ target["gt_rotmats"][offset, 0]
            torso_error = (
                (pred_body0[0].transpose(0, 1) @ pred_torso)
                - (target["gt_rotmats"][0, 0].transpose(0, 1) @ gt_torso)
            ).square().mean()
            terms["torso"].append(torso_error / scales["human_torso"])
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
        balanced["camera_t"]
        + balanced["camera_r"]
        + balanced["pointmap"]
        + 0.2 * balanced["depth"]
        + 0.3 * balanced["human_root"]
        + 0.05 * balanced["torso"]
    )
    return total, {key: float(value.detach()) for key, value in means.items()}


def distillation_auxiliary(output, pair: dict, device: torch.device, variant: str) -> torch.Tensor:
    fresh = pair["fresh_state"].to(device).float().unsqueeze(0)
    oracle_residual = pair["oracle_residual"].to(device).float().unsqueeze(0)
    gate_target = torch.tensor([pair["labels"]["gate_target"]], device=device)
    desired_gate = torch.ones_like(gate_target) if variant == "ungated" else gate_target
    desired = fresh + desired_gate[:, None, None] * oracle_residual
    energy = oracle_residual.square().mean().clamp_min(1e-4)
    latent = (output.corrected_state - desired).square().mean() / energy
    gain_target = torch.tensor([pair["labels"]["gain_target"]], device=device).clamp(-1.0, 1.0)
    gain = F.smooth_l1_loss(output.predicted_gain, gain_target)
    if variant == "ungated":
        gate = latent.new_zeros(())
    else:
        gate = F.mse_loss(output.gate, gate_target)
    return 0.20 * latent + 0.10 * gate + 0.05 * gain
