"""Bounded gated first-write adapter for the V12 gauge-neutral probe."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GatedFirstWriteOutput:
    corrected_state: torch.Tensor
    bounded_residual: torch.Tensor
    gate: torch.Tensor
    gate_logit: torch.Tensor
    predicted_gain: torch.Tensor
    wait_score: torch.Tensor


class GatedGaugeNeutralFirstWritePrompt(nn.Module):
    """Read old context and produce a bounded residual for the first fresh write."""

    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 192,
        image_summary_dim: int = 2048,
        human_summary_dim: int = 1536,
        camera_dim: int = 768,
        memory_summary_dim: int = 3072,
        diagnostic_dim: int = 8,
        max_residual_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.diagnostic_dim = int(diagnostic_dim)
        self.max_residual_std = float(max_residual_std)

        self.old_state_norm = nn.LayerNorm(self.state_dim)
        self.fresh_state_norm = nn.LayerNorm(self.state_dim)
        self.old_state_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.fresh_state_proj = nn.Linear(self.state_dim, self.hidden_dim)

        self.context_branches = nn.ModuleList(
            [
                self._branch(self.state_dim * 2),
                self._branch(image_summary_dim),
                self._branch(human_summary_dim),
                self._branch(camera_dim),
                self._branch(memory_summary_dim),
                self._branch(self.diagnostic_dim),
            ]
        )
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * len(self.context_branches)),
            nn.Linear(self.hidden_dim * len(self.context_branches), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.residual_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.wait_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.5)

    def _branch(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def mean_std(tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=-1)

    def forward(
        self,
        old_state: torch.Tensor,
        fresh_state: torch.Tensor,
        image_summary: torch.Tensor,
        human_summary: torch.Tensor,
        camera_token: torch.Tensor,
        memory_summary: torch.Tensor,
        diagnostics: torch.Tensor | None = None,
        gate_override: float | torch.Tensor | None = None,
    ) -> GatedFirstWriteOutput:
        if diagnostics is None:
            diagnostics = fresh_state.new_zeros(fresh_state.shape[0], self.diagnostic_dim)
        contexts = (
            self.mean_std(old_state),
            image_summary,
            human_summary,
            camera_token,
            memory_summary,
            diagnostics,
        )
        context = self.context_fuse(
            torch.cat(
                [branch(value.float()) for branch, value in zip(self.context_branches, contexts)],
                dim=-1,
            )
        )
        token_hidden = self.old_state_proj(self.old_state_norm(old_state.float()))
        token_hidden = token_hidden + self.fresh_state_proj(self.fresh_state_norm(fresh_state.float()))
        token_hidden = self.token_fuse(token_hidden + context[:, None, :])
        raw_residual = self.residual_head(token_hidden)
        state_scale = fresh_state.float().std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-4)
        bounded_residual = self.max_residual_std * state_scale * torch.tanh(raw_residual)
        gate_logit = self.gate_head(context).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        if gate_override is not None:
            if isinstance(gate_override, torch.Tensor):
                gate = gate_override.to(device=gate.device, dtype=gate.dtype).reshape_as(gate)
            else:
                gate = torch.full_like(gate, float(gate_override))
        corrected_state = fresh_state.float() + gate[:, None, None] * bounded_residual
        return GatedFirstWriteOutput(
            corrected_state=corrected_state,
            bounded_residual=bounded_residual,
            gate=gate,
            gate_logit=gate_logit,
            predicted_gain=self.gain_head(context).squeeze(-1),
            wait_score=torch.sigmoid(self.wait_head(context).squeeze(-1)),
        )
