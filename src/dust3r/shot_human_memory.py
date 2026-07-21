"""Deterministic cross-shot human memory for V14.1 streaming probes.

The memory is deliberately isolated from Human3R's recurrent scene state.  It
can condition decoded human tokens and stabilize gauge-invariant SMPL-X fields,
but it never changes camera, image, pointmap, or pose-memory tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _as_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


@dataclass(frozen=True)
class HumanMemoryConfig:
    token_alpha: float = 0.0
    shape_alpha: float = 0.0
    local_pose_alpha: float = 0.0
    update_alpha: float = 0.2
    attention_temperature: float = 0.15
    commit_mode: str = "align"
    min_detection_score: float = 0.35
    max_shape_delta: float = 3.0
    max_world_root_jump: float = 1.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "HumanMemoryConfig":
        values = {} if values is None else values
        return cls(
            token_alpha=_as_float(values.get("token_alpha"), cls.token_alpha),
            shape_alpha=_as_float(values.get("shape_alpha"), cls.shape_alpha),
            local_pose_alpha=_as_float(values.get("local_pose_alpha"), cls.local_pose_alpha),
            update_alpha=_as_float(values.get("update_alpha"), cls.update_alpha),
            attention_temperature=_as_float(
                values.get("attention_temperature"), cls.attention_temperature
            ),
            commit_mode=str(values.get("commit_mode", cls.commit_mode)),
            min_detection_score=_as_float(
                values.get("min_detection_score"), cls.min_detection_score
            ),
            max_shape_delta=_as_float(values.get("max_shape_delta"), cls.max_shape_delta),
            max_world_root_jump=_as_float(
                values.get("max_world_root_jump"), cls.max_world_root_jump
            ),
        )


def project_to_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3x3 matrices to SO(3)."""

    u, _, vh = torch.linalg.svd(matrix.float())
    rotation = u @ vh
    determinant = torch.det(rotation)
    correction = torch.zeros_like(matrix)
    correction[..., 0, 0] = 1.0
    correction[..., 1, 1] = 1.0
    correction[..., 2, 2] = torch.where(
        determinant < 0.0,
        determinant.new_tensor(-1.0),
        determinant.new_tensor(1.0),
    )
    return (u @ correction @ vh).to(dtype=matrix.dtype)


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bnj->bni", transform[:, :3, :3], points) + transform[:, None, :3, 3]


class StreamingHumanMemory:
    """Small, training-free human memory bank used only by V14.1."""

    def __init__(self, config: HumanMemoryConfig, override: Mapping[str, Any] | None = None):
        self.config = config
        self.token: torch.Tensor | None = None
        self.shape: torch.Tensor | None = None
        self.local_rotmat: torch.Tensor | None = None
        self.world_root: torch.Tensor | None = None
        self.commit_count = 0
        self.override = None if override is None else dict(override)

    @property
    def populated(self) -> bool:
        return self.token is not None

    def clear(self) -> None:
        self.token = None
        self.shape = None
        self.local_rotmat = None
        self.world_root = None
        self.commit_count = 0

    def load_override(self, reference: torch.Tensor) -> None:
        if self.override is None:
            return
        for name in ("token", "shape", "local_rotmat", "world_root"):
            value = self.override.get(name)
            if value is None:
                continue
            value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            if value.ndim == reference.ndim - 1:
                value = value.unsqueeze(0)
            setattr(self, name, value.detach().clone())
        self.override = None

    def attention(self, current_token: torch.Tensor) -> torch.Tensor | None:
        if self.token is None or self.token.numel() == 0 or current_token.numel() == 0:
            return None
        query = F.normalize(current_token.float(), dim=-1, eps=1e-6)
        key = F.normalize(self.token.to(current_token).float(), dim=-1, eps=1e-6)
        logits = torch.einsum("bnd,bmd->bnm", query, key)
        logits = logits / max(self.config.attention_temperature, 1e-4)
        return torch.softmax(logits, dim=-1).to(dtype=current_token.dtype)

    @staticmethod
    def _read_bank(weights: torch.Tensor | None, bank: torch.Tensor | None) -> torch.Tensor | None:
        if weights is None or bank is None:
            return None
        flat = bank.reshape(bank.shape[0], bank.shape[1], -1)
        context = torch.einsum("bnm,bmd->bnd", weights, flat)
        return context.reshape(weights.shape[0], weights.shape[1], *bank.shape[2:])

    def condition_token(
        self, current_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self.load_override(current_token)
        weights = self.attention(current_token)
        context = self._read_bank(weights, self.token)
        if context is None or self.config.token_alpha <= 0.0:
            return current_token, weights, context
        alpha = min(max(self.config.token_alpha, 0.0), 1.0)
        conditioned = current_token + alpha * (context.to(current_token) - current_token)
        return conditioned, weights, context

    def stabilize_prediction(
        self,
        prediction: dict[str, torch.Tensor],
        weights: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        diagnostics: dict[str, torch.Tensor] = {}
        shape = prediction.get("smpl_shape")
        shape_context = self._read_bank(weights, self.shape)
        if shape is not None and shape_context is not None and self.config.shape_alpha > 0.0:
            diagnostics["shape_raw"] = shape.detach().clone()
            alpha = min(max(self.config.shape_alpha, 0.0), 1.0)
            prediction["smpl_shape"] = shape + alpha * (shape_context.to(shape) - shape)

        rotmat = prediction.get("smpl_rotmat")
        pose_context = self._read_bank(weights, self.local_rotmat)
        if (
            rotmat is not None
            and rotmat.shape[-3] > 1
            and pose_context is not None
            and self.config.local_pose_alpha > 0.0
        ):
            diagnostics["rotmat_raw"] = rotmat.detach().clone()
            alpha = min(max(self.config.local_pose_alpha, 0.0), 1.0)
            local = rotmat[..., 1:, :, :]
            blended = local + alpha * (pose_context.to(local) - local)
            prediction["smpl_rotmat"] = torch.cat(
                [rotmat[..., :1, :, :], project_to_rotation(blended)], dim=-3
            )
        return diagnostics

    def quality(
        self,
        prediction: Mapping[str, torch.Tensor],
        detection_score: torch.Tensor | None,
        world_root: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shape = prediction.get("smpl_shape")
        if shape is None:
            reference = detection_score if detection_score is not None else world_root
            if reference is None:
                raise ValueError("Human-memory quality requires a human prediction")
            valid = torch.zeros(reference.shape[:2], device=reference.device, dtype=torch.bool)
            return valid, {"valid": valid.float()}

        valid = torch.isfinite(shape).all(dim=-1)
        diagnostics: dict[str, torch.Tensor] = {}
        if detection_score is not None:
            score = detection_score.reshape(shape.shape[0], -1)[:, : shape.shape[1]]
            valid = valid & (score >= self.config.min_detection_score)
            diagnostics["detection_score"] = score.detach()
        if self.shape is not None:
            shape_weights = weights
            if shape_weights is None and self.shape.shape[:2] == shape.shape[:2]:
                shape_weights = torch.eye(
                    shape.shape[1], device=shape.device, dtype=shape.dtype
                ).unsqueeze(0).expand(shape.shape[0], -1, -1)
            if shape_weights is not None:
                reference_shape = self._read_bank(shape_weights, self.shape)
                shape_delta = torch.linalg.vector_norm(shape - reference_shape.to(shape), dim=-1)
                valid = valid & (shape_delta <= self.config.max_shape_delta)
                diagnostics["shape_delta"] = shape_delta.detach()
        if world_root is not None:
            valid = valid & torch.isfinite(world_root).all(dim=-1)
            if self.world_root is not None and self.world_root.shape[1] == world_root.shape[1]:
                jump = torch.linalg.vector_norm(world_root - self.world_root.to(world_root), dim=-1)
                valid = valid & (jump <= self.config.max_world_root_jump)
                diagnostics["world_root_jump"] = jump.detach()
        diagnostics["valid"] = valid.float()
        return valid, diagnostics

    def commit(
        self,
        token: torch.Tensor,
        prediction: Mapping[str, torch.Tensor],
        world_root: torch.Tensor | None,
        valid: torch.Tensor | None = None,
    ) -> None:
        shape = prediction.get("smpl_shape")
        rotmat = prediction.get("smpl_rotmat")
        local_rotmat = None if rotmat is None or rotmat.shape[-3] <= 1 else rotmat[..., 1:, :, :]
        if valid is not None and not bool(valid.any().item()):
            return

        alpha = min(max(self.config.update_alpha, 0.0), 1.0)
        if self.token is None or self.token.shape != token.shape:
            self.token = token.detach().clone()
        else:
            self.token = self.token + alpha * (token.detach() - self.token)
        if shape is not None:
            if self.shape is None or self.shape.shape != shape.shape:
                self.shape = shape.detach().clone()
            else:
                self.shape = self.shape + alpha * (shape.detach() - self.shape)
        if local_rotmat is not None:
            if self.local_rotmat is None or self.local_rotmat.shape != local_rotmat.shape:
                self.local_rotmat = local_rotmat.detach().clone()
            else:
                blended = self.local_rotmat + alpha * (local_rotmat.detach() - self.local_rotmat)
                self.local_rotmat = project_to_rotation(blended)
        if world_root is not None:
            if self.world_root is None or self.world_root.shape != world_root.shape:
                self.world_root = world_root.detach().clone()
            else:
                self.world_root = self.world_root + alpha * (world_root.detach() - self.world_root)
        self.commit_count += 1

    def snapshot(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name in ("token", "shape", "local_rotmat", "world_root"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value.detach().clone()
        return result
