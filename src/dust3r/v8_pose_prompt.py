"""V8.1 UniCon-style pose correction prompt modules.

These modules are intentionally small and decoder-agnostic. They build a
pose-correction prompt before the current-frame decoder, then predict a latent
residual from the refined prompt after decoder attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class V81PosePromptOutput:
    corr_tokens: torch.Tensor
    body_tokens: torch.Tensor
    history_token: torch.Tensor
    camera_motion_token: torch.Tensor
    reliability_token: torch.Tensor
    body_attention: Optional[torch.Tensor] = None


def make_v8_corr_pos(batch_size, num_tokens, device, dtype):
    return -torch.ones(batch_size, num_tokens, 2, device=device, dtype=dtype)


class V81PoseCorrectionPrompt(nn.Module):
    """Build A_corr_t before the decoder.

    First version uses four prompt sources:
      1. body-part queries reading current image/human tokens,
      2. previous prompt/pose history,
      3. current coarse pose token plus pose memory,
      4. a lightweight reliability token.
    """

    def __init__(
        self,
        enc_dim: int,
        dec_dim: int,
        num_body_queries: int = 4,
        num_heads: int = 8,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.dec_dim = int(dec_dim)
        self.num_body_queries = int(num_body_queries)
        self.num_corr_tokens = 4
        self.memory_dim = int(memory_dim or dec_dim * 2)

        self.image_proj = nn.Linear(self.enc_dim, self.dec_dim)
        self.human_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.state_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.memory_proj = nn.Linear(self.memory_dim, self.dec_dim)

        self.body_queries = nn.Parameter(torch.randn(1, self.num_body_queries, self.dec_dim) * 0.02)
        self.no_history_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        self.token_type_embed = nn.Parameter(torch.randn(1, self.num_corr_tokens, self.dec_dim) * 0.02)

        self.query_norm = nn.LayerNorm(self.dec_dim)
        self.context_norm = nn.LayerNorm(self.dec_dim)
        self.body_attention = nn.MultiheadAttention(
            self.dec_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.history_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 2),
            nn.Linear(self.dec_dim * 2, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.dec_dim),
        )
        self.camera_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 3),
            nn.Linear(self.dec_dim * 3, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.dec_dim),
        )
        self.reliability_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 3),
            nn.Linear(self.dec_dim * 3, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.dec_dim),
        )
        self.out_norm = nn.LayerNorm(self.dec_dim)

    def _mean_or_zero(self, tokens: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if tokens is None or tokens.numel() == 0 or tokens.shape[1] == 0:
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        return tokens.mean(dim=1, keepdim=True)

    def _project_memory(self, pose_memory: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if pose_memory is None or pose_memory.numel() == 0 or pose_memory.shape[1] == 0:
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        return self.memory_proj(pose_memory).mean(dim=1, keepdim=True)

    def forward(
        self,
        image_tokens: torch.Tensor,
        pose_token: torch.Tensor,
        human_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
        pose_memory: Optional[torch.Tensor] = None,
        prev_corr_token: Optional[torch.Tensor] = None,
        prev_pose_token: Optional[torch.Tensor] = None,
        prev_gate: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> V81PosePromptOutput:
        batch_size = image_tokens.shape[0]
        image_ctx = self.image_proj(image_tokens)
        contexts = [image_ctx]
        if human_tokens is not None and human_tokens.numel() > 0 and human_tokens.shape[1] > 0:
            contexts.append(self.human_proj(human_tokens))
        context = self.context_norm(torch.cat(contexts, dim=1))

        body_query = self.body_queries.expand(batch_size, -1, -1)
        body_tokens, body_attn = self.body_attention(
            self.query_norm(body_query),
            context,
            context,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        body_summary = body_tokens.mean(dim=1, keepdim=True)

        if prev_corr_token is None:
            prev_corr = self.no_history_token.expand(batch_size, -1, -1)
        else:
            prev_corr = self._mean_or_zero(prev_corr_token, batch_size, pose_token)
        prev_pose = pose_token if prev_pose_token is None else self._mean_or_zero(prev_pose_token, batch_size, pose_token)
        history_token = self.history_mlp(torch.cat([prev_corr, prev_pose], dim=-1))

        state_summary = self._mean_or_zero(
            self.state_proj(state_tokens) if state_tokens is not None else None,
            batch_size,
            pose_token,
        )
        memory_summary = self._project_memory(pose_memory, batch_size, pose_token)
        camera_motion_token = self.camera_mlp(torch.cat([pose_token, memory_summary, prev_pose], dim=-1))

        if prev_gate is None:
            gate_token = pose_token.new_zeros(batch_size, 1, self.dec_dim)
        else:
            gate_value = prev_gate.reshape(batch_size, 1, 1).to(dtype=pose_token.dtype, device=pose_token.device)
            gate_token = gate_value.expand(-1, -1, self.dec_dim)
        reliability_token = self.reliability_mlp(torch.cat([body_summary, state_summary, gate_token], dim=-1))

        corr_tokens = torch.cat(
            [body_summary, history_token, camera_motion_token, reliability_token],
            dim=1,
        )
        corr_tokens = self.out_norm(corr_tokens + self.token_type_embed)

        return V81PosePromptOutput(
            corr_tokens=corr_tokens,
            body_tokens=body_tokens,
            history_token=history_token,
            camera_motion_token=camera_motion_token,
            reliability_token=reliability_token,
            body_attention=body_attn if return_attention else None,
        )


class V81PoseLatentResidualHead(nn.Module):
    """Predict a latent residual for the refined pose token."""

    def __init__(self, dec_dim: int, hidden_dim: Optional[int] = None, gate_bias: float = -4.0):
        super().__init__()
        self.dec_dim = int(dec_dim)
        hidden = int(hidden_dim or dec_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.dec_dim),
            nn.Linear(self.dec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dec_dim),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.dec_dim),
            nn.Linear(self.dec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)

    def forward(self, refined_corr_tokens: torch.Tensor):
        pooled = refined_corr_tokens.mean(dim=1)
        delta = self.delta_head(pooled).unsqueeze(1)
        gate = torch.sigmoid(self.gate_head(pooled)).unsqueeze(1)
        corrected_delta = gate * delta
        return corrected_delta, gate, delta
