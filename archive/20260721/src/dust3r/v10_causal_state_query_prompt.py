"""Small read-old/write-fresh adapter used by the V10 causal probe.

The module never owns or updates the old Human3R state.  It reads the old
state together with current-frame summaries and predicts a residual for the
fresh branch's first state write.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CausalStateQueryOutput:
    corrected_state: torch.Tensor
    state_delta: torch.Tensor
    state_gate: torch.Tensor
    predicted_difficulty: torch.Tensor


class CausalStateQueryFirstWritePrompt(nn.Module):
    """Token-wise low-rank guidance for the first fresh-state write."""

    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 256,
        image_summary_dim: int = 2048,
        human_summary_dim: int = 1536,
        camera_dim: int = 768,
        memory_summary_dim: int = 3072,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.old_state_norm = nn.LayerNorm(self.state_dim)
        self.fresh_state_norm = nn.LayerNorm(self.state_dim)
        self.old_state_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.fresh_state_proj = nn.Linear(self.state_dim, self.hidden_dim)

        self.old_summary_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim * 2),
            nn.Linear(self.state_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_summary_dim),
            nn.Linear(image_summary_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.human_proj = nn.Sequential(
            nn.LayerNorm(human_summary_dim),
            nn.Linear(human_summary_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.camera_proj = nn.Sequential(
            nn.LayerNorm(camera_dim),
            nn.Linear(camera_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(memory_summary_dim),
            nn.Linear(memory_summary_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 5),
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.difficulty_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -0.5)

    @staticmethod
    def _mean_std(tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)],
            dim=-1,
        )

    def forward(
        self,
        old_state: torch.Tensor,
        fresh_state: torch.Tensor,
        image_summary: torch.Tensor,
        human_summary: torch.Tensor,
        camera_token: torch.Tensor,
        memory_summary: torch.Tensor,
    ) -> CausalStateQueryOutput:
        old_summary = self._mean_std(old_state)
        context = self.context_fuse(
            torch.cat(
                [
                    self.old_summary_proj(old_summary),
                    self.image_proj(image_summary),
                    self.human_proj(human_summary),
                    self.camera_proj(camera_token),
                    self.memory_proj(memory_summary),
                ],
                dim=-1,
            )
        )
        token_hidden = self.old_state_proj(self.old_state_norm(old_state))
        token_hidden = token_hidden + self.fresh_state_proj(self.fresh_state_norm(fresh_state))
        token_hidden = self.token_fuse(token_hidden + context[:, None, :])
        state_delta = self.delta_head(token_hidden)
        state_gate = torch.sigmoid(self.gate_head(token_hidden))
        corrected_state = fresh_state + state_gate * state_delta
        predicted_difficulty = self.difficulty_head(context).squeeze(-1)
        return CausalStateQueryOutput(
            corrected_state=corrected_state,
            state_delta=state_delta,
            state_gate=state_gate,
            predicted_difficulty=predicted_difficulty,
        )
