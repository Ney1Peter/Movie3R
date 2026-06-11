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


def _has_aligned_batch(tokens: Optional[torch.Tensor], batch_size: int) -> bool:
    return (
        tokens is not None
        and tokens.ndim >= 3
        and tokens.numel() > 0
        and tokens.shape[0] == batch_size
        and tokens.shape[1] > 0
    )


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
        use_history: bool = True,
        use_pose_memory: bool = True,
        use_reliability: bool = True,
    ):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.dec_dim = int(dec_dim)
        self.num_body_queries = int(num_body_queries)
        self.num_corr_tokens = 4
        self.memory_dim = int(memory_dim or dec_dim * 2)
        self.use_history = bool(use_history)
        self.use_pose_memory = bool(use_pose_memory)
        self.use_reliability = bool(use_reliability)

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
        if not _has_aligned_batch(tokens, batch_size):
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        return tokens.mean(dim=1, keepdim=True)

    def _project_memory(self, pose_memory: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if not _has_aligned_batch(pose_memory, batch_size):
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
        prev_delta_token: Optional[torch.Tensor] = None,
        prev_gate: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> V81PosePromptOutput:
        batch_size = image_tokens.shape[0]
        image_ctx = self.image_proj(image_tokens)
        contexts = [image_ctx]
        # If a tokenizer returns unbatched detections, do not share them across samples.
        if _has_aligned_batch(human_tokens, batch_size):
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

        if not self.use_history:
            prev_corr = pose_token.new_zeros(batch_size, 1, self.dec_dim)
            prev_pose = pose_token.new_zeros(batch_size, 1, self.dec_dim)
        elif prev_corr_token is None:
            prev_corr = self.no_history_token.expand(batch_size, -1, -1)
            prev_pose = pose_token if prev_pose_token is None else self._mean_or_zero(prev_pose_token, batch_size, pose_token)
        else:
            prev_corr = self._mean_or_zero(prev_corr_token, batch_size, pose_token)
            prev_pose = pose_token if prev_pose_token is None else self._mean_or_zero(prev_pose_token, batch_size, pose_token)
        history_token = self.history_mlp(torch.cat([prev_corr, prev_pose], dim=-1))

        if self.use_pose_memory:
            memory_summary = self._project_memory(pose_memory, batch_size, pose_token)
            camera_motion_token = self.camera_mlp(torch.cat([pose_token, memory_summary, prev_pose], dim=-1))
        else:
            memory_summary = pose_token.new_zeros(batch_size, 1, self.dec_dim)
            camera_motion_token = pose_token.new_zeros(batch_size, 1, self.dec_dim)

        if prev_gate is None or prev_gate.reshape(-1).numel() != batch_size or not self.use_reliability:
            gate_token = pose_token.new_zeros(batch_size, 1, self.dec_dim)
        else:
            gate_value = prev_gate.reshape(-1).reshape(batch_size, 1, 1).to(
                dtype=pose_token.dtype, device=pose_token.device
            )
            gate_token = gate_value.expand(-1, -1, self.dec_dim)
        if self.use_reliability:
            state_summary = self._mean_or_zero(
                self.state_proj(state_tokens) if state_tokens is not None else None,
                batch_size,
                pose_token,
            )
            reliability_token = self.reliability_mlp(torch.cat([body_summary, state_summary, gate_token], dim=-1))
        else:
            reliability_token = pose_token.new_zeros(batch_size, 1, self.dec_dim)

        token_items = [(0, body_summary)]
        if self.use_history:
            token_items.append((1, history_token))
        if self.use_pose_memory:
            token_items.append((2, camera_motion_token))
        if self.use_reliability:
            token_items.append((3, reliability_token))
        corr_tokens = torch.cat(
            [token + self.token_type_embed[:, type_idx:type_idx + 1] for type_idx, token in token_items],
            dim=1,
        )
        corr_tokens = self.out_norm(corr_tokens)

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

    def __init__(self, dec_dim: int, hidden_dim: Optional[int] = None, gate_bias: float = -4.0, use_gate: bool = True):
        super().__init__()
        self.dec_dim = int(dec_dim)
        self.use_gate = bool(use_gate)
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
        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(pooled)).unsqueeze(1)
        else:
            gate = torch.ones(delta.shape[0], 1, 1, device=delta.device, dtype=delta.dtype)
        corrected_delta = gate * delta
        return corrected_delta, gate, delta


class V82PoseRelationPrompt(nn.Module):
    """Build the V8.2 pose-relation prompt before decoder attention.

    This follows the UniCon3R grouping more closely than the V8.1 body-query
    prompt. The prompt contains three relation tokens:
      1. semantic pose-scene context from current tokens and recurrent memory,
      2. explicit latent alignment cue from current/previous pose tokens,
      3. temporal momentum from previous correction token/residual/gate.
    """

    def __init__(
        self,
        enc_dim: int,
        dec_dim: int,
        num_heads: int = 8,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_history: bool = True,
        use_pose_memory: bool = True,
        use_reliability: bool = True,
    ):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.dec_dim = int(dec_dim)
        self.num_corr_tokens = 3
        self.memory_dim = int(memory_dim or dec_dim * 2)
        self.use_history = bool(use_history)
        self.use_pose_memory = bool(use_pose_memory)
        self.use_reliability = bool(use_reliability)

        self.image_proj = nn.Linear(self.enc_dim, self.dec_dim)
        self.human_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.state_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.memory_proj = nn.Linear(self.memory_dim, self.dec_dim)

        self.relation_query = nn.Parameter(torch.randn(1, 1, self.dec_dim) * 0.02)
        self.no_history_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        self.token_type_embed = nn.Parameter(torch.randn(1, self.num_corr_tokens, self.dec_dim) * 0.02)

        self.current_norm = nn.LayerNorm(self.dec_dim)
        self.memory_norm = nn.LayerNorm(self.dec_dim)
        self.query_norm = nn.LayerNorm(self.dec_dim)
        self.current_attention = nn.MultiheadAttention(
            self.dec_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_attention = nn.MultiheadAttention(
            self.dec_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_gate = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 2),
            nn.Linear(self.dec_dim * 2, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, 1),
        )
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.zeros_(self.semantic_gate[-1].bias)

        self.alignment_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 4),
            nn.Linear(self.dec_dim * 4, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.dec_dim),
        )
        self.momentum_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 3),
            nn.Linear(self.dec_dim * 3, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.dec_dim),
        )
        self.out_norm = nn.LayerNorm(self.dec_dim)

    def _mean_or_zero(self, tokens: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if not _has_aligned_batch(tokens, batch_size):
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        return tokens.mean(dim=1, keepdim=True)

    def _project_memory(self, pose_memory: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if (
            not _has_aligned_batch(pose_memory, batch_size)
            or not self.use_pose_memory
        ):
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        return self.memory_proj(pose_memory)

    def _gate_token(self, prev_gate: Optional[torch.Tensor], batch_size: int, ref: torch.Tensor):
        if prev_gate is None or prev_gate.reshape(-1).numel() != batch_size or not self.use_reliability:
            return ref.new_zeros(batch_size, 1, self.dec_dim)
        gate_value = prev_gate.reshape(-1).reshape(batch_size, 1, 1).to(device=ref.device, dtype=ref.dtype)
        return gate_value.expand(-1, -1, self.dec_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        pose_token: torch.Tensor,
        human_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
        pose_memory: Optional[torch.Tensor] = None,
        prev_corr_token: Optional[torch.Tensor] = None,
        prev_pose_token: Optional[torch.Tensor] = None,
        prev_delta_token: Optional[torch.Tensor] = None,
        prev_gate: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> V81PosePromptOutput:
        batch_size = image_tokens.shape[0]
        query = self.query_norm(self.relation_query.expand(batch_size, -1, -1))

        current_parts = [self.image_proj(image_tokens), pose_token]
        # Keep image-only batch training sample-local: mismatched human tokens
        # cannot be assigned safely to batch elements.
        if _has_aligned_batch(human_tokens, batch_size):
            current_parts.append(self.human_proj(human_tokens))
        current_context = self.current_norm(torch.cat(current_parts, dim=1))
        current_token, current_attn = self.current_attention(
            query,
            current_context,
            current_context,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        memory_parts = []
        if _has_aligned_batch(state_tokens, batch_size):
            memory_parts.append(self.state_proj(state_tokens))
        projected_memory = self._project_memory(pose_memory, batch_size, pose_token)
        if projected_memory.numel() > 0 and projected_memory.shape[1] > 0:
            memory_parts.append(projected_memory)
        if _has_aligned_batch(prev_corr_token, batch_size) and self.use_history:
            memory_parts.append(prev_corr_token)
        if memory_parts:
            memory_context = self.memory_norm(torch.cat(memory_parts, dim=1))
        else:
            memory_context = self.no_history_token.expand(batch_size, -1, -1)
        memory_token, _ = self.memory_attention(query, memory_context, memory_context, need_weights=False)

        semantic_gamma = torch.sigmoid(self.semantic_gate(torch.cat([current_token, memory_token], dim=-1)))
        semantic_token = semantic_gamma * current_token + (1.0 - semantic_gamma) * memory_token

        if self.use_history and prev_pose_token is not None and prev_pose_token.numel() > 0:
            prev_pose = self._mean_or_zero(prev_pose_token, batch_size, pose_token)
        else:
            prev_pose = pose_token
        pose_delta_latent = pose_token - prev_pose
        alignment_token = self.alignment_mlp(
            torch.cat([pose_token, prev_pose, pose_delta_latent, memory_token], dim=-1)
        )

        if self.use_history:
            prev_corr = self._mean_or_zero(prev_corr_token, batch_size, pose_token)
            prev_delta = self._mean_or_zero(prev_delta_token, batch_size, pose_token)
            prev_gate_token = self._gate_token(prev_gate, batch_size, pose_token)
            momentum_token = self.momentum_mlp(torch.cat([prev_corr, prev_delta, prev_gate_token], dim=-1))
        else:
            momentum_token = pose_token.new_zeros(batch_size, 1, self.dec_dim)

        corr_tokens = torch.cat(
            [
                semantic_token + self.token_type_embed[:, 0:1],
                alignment_token + self.token_type_embed[:, 1:2],
                momentum_token + self.token_type_embed[:, 2:3],
            ],
            dim=1,
        )
        corr_tokens = self.out_norm(corr_tokens)

        return V81PosePromptOutput(
            corr_tokens=corr_tokens,
            body_tokens=current_token,
            history_token=momentum_token,
            camera_motion_token=alignment_token,
            reliability_token=semantic_token,
            body_attention=current_attn if return_attention else None,
        )


class V82PoseRelationResidualHead(nn.Module):
    """Predict pose-latent residual and drift-bound gate from refined A_corr_t."""

    def __init__(
        self,
        dec_dim: int,
        hidden_dim: Optional[int] = None,
        gate_bias: float = 0.0,
        use_gate: bool = True,
    ):
        super().__init__()
        self.dec_dim = int(dec_dim)
        self.use_gate = bool(use_gate)
        hidden = int(hidden_dim or dec_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.dec_dim),
            nn.Linear(self.dec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dec_dim),
        )
        self.drift_head = nn.Sequential(
            nn.LayerNorm(self.dec_dim),
            nn.Linear(self.dec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.drift_head[-1].weight)
        nn.init.constant_(self.drift_head[-1].bias, gate_bias)

    def forward(self, refined_corr_tokens: torch.Tensor):
        pooled = refined_corr_tokens.mean(dim=1)
        delta = self.delta_head(pooled).unsqueeze(1)
        drift_logit = self.drift_head(pooled).unsqueeze(1)
        if self.use_gate:
            gate = torch.sigmoid(drift_logit)
        else:
            gate = torch.ones(delta.shape[0], 1, 1, device=delta.device, dtype=delta.dtype)
        corrected_delta = gate * delta
        return corrected_delta, gate, delta, drift_logit


class V8HumanTranslationCorrectionHead(nn.Module):
    """Explicit sanity-check head for correcting whole-body SMPL translation.

    This is intentionally a small post-human-head residual. It tests whether the
    refined V8 correction token already contains enough human alignment signal
    before moving the idea into a fully latent UniCon-style human branch.
    """

    def __init__(
        self,
        dec_dim: int,
        hidden_dim: Optional[int] = None,
        gate_bias: float = 0.0,
        use_gate: bool = True,
        max_delta: Optional[float] = None,
        gate_mode: str = "independent",
    ):
        super().__init__()
        self.dec_dim = int(dec_dim)
        self.use_gate = bool(use_gate)
        self.max_delta = None if max_delta is None else float(max_delta)
        self.gate_mode = str(gate_mode)
        hidden = int(hidden_dim or dec_dim)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 3),
            nn.Linear(self.dec_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dec_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.dec_dim, 3)
        self.gate_head = nn.Linear(self.dec_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_bias)

    def forward(
        self,
        human_tokens: torch.Tensor,
        corr_tokens: torch.Tensor,
        pose_token: torch.Tensor,
        smpl_transl: torch.Tensor,
        shared_gate: Optional[torch.Tensor] = None,
        apply_mask: Optional[torch.Tensor] = None,
    ):
        if human_tokens is None or human_tokens.numel() == 0:
            return smpl_transl, None
        if corr_tokens is None or corr_tokens.numel() == 0:
            return smpl_transl, None
        batch_size, num_humans = human_tokens.shape[:2]
        if smpl_transl.shape[0] != batch_size or smpl_transl.shape[1] == 0:
            return smpl_transl, None

        num_apply = min(num_humans, smpl_transl.shape[1])
        human_context = human_tokens[:, :num_apply]
        corr_context = corr_tokens.mean(dim=1, keepdim=True).expand(-1, num_apply, -1)
        pose_context = pose_token.mean(dim=1, keepdim=True).expand(-1, num_apply, -1)
        context = torch.cat([human_context, corr_context, pose_context], dim=-1)
        hidden = self.context_mlp(context)
        delta_raw = self.delta_head(hidden)
        if self.max_delta is not None and self.max_delta > 0:
            delta = self.max_delta * torch.tanh(delta_raw)
        else:
            delta = delta_raw
        if self.use_gate:
            learned_gate = torch.sigmoid(self.gate_head(hidden))
        else:
            learned_gate = torch.ones_like(delta[..., :1])
        gate = learned_gate
        shared_gate_used = None
        if shared_gate is not None:
            shared_gate_used = shared_gate.to(device=delta.device, dtype=delta.dtype).reshape(batch_size, -1, 1)
            if shared_gate_used.shape[1] == 1:
                shared_gate_used = shared_gate_used.expand(-1, num_apply, -1)
            else:
                shared_gate_used = shared_gate_used[:, :num_apply]
            if self.gate_mode == "shared":
                gate = shared_gate_used
            elif self.gate_mode == "product":
                gate = shared_gate_used * learned_gate
        if apply_mask is not None:
            gate = gate * apply_mask.to(device=gate.device, dtype=gate.dtype)
        delta_applied = gate * delta

        corrected = smpl_transl.clone()
        corrected[:, :num_apply] = corrected[:, :num_apply] + delta_applied
        info = {
            "v8_human_trans_corr_delta_raw": delta_raw,
            "v8_human_trans_corr_delta_applied": delta_applied,
            "v8_human_trans_corr_gate": gate,
            "v8_human_trans_corr_learned_gate": learned_gate,
            "v8_human_trans_corr_delta_norm": delta_raw.norm(dim=-1),
        }
        if shared_gate_used is not None:
            info["v8_human_trans_corr_shared_gate"] = shared_gate_used
        return corrected, info


class V8HumanLatentResidualHead(nn.Module):
    """Correct the decoder human token before the original Human3R human head.

    Unlike :class:`V8HumanTranslationCorrectionHead`, this head never writes a
    SMPL parameter directly. It predicts a residual in the Human3R decoder
    human-token space, then the frozen human head interprets the corrected token
    as SMPL parameters.
    """

    def __init__(
        self,
        dec_dim: int,
        hidden_dim: Optional[int] = None,
        gate_bias: float = 0.0,
        use_gate: bool = True,
        max_delta: Optional[float] = None,
        gate_mode: str = "shared",
    ):
        super().__init__()
        self.dec_dim = int(dec_dim)
        self.use_gate = bool(use_gate)
        self.max_delta = None if max_delta is None else float(max_delta)
        self.gate_mode = str(gate_mode)
        hidden = int(hidden_dim or dec_dim)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(self.dec_dim * 3),
            nn.Linear(self.dec_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dec_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.dec_dim, self.dec_dim)
        self.gate_head = nn.Linear(self.dec_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_bias)

    def forward(
        self,
        human_tokens: torch.Tensor,
        corr_tokens: torch.Tensor,
        pose_token: torch.Tensor,
        shared_gate: Optional[torch.Tensor] = None,
        apply_mask: Optional[torch.Tensor] = None,
    ):
        if human_tokens is None or human_tokens.numel() == 0:
            return human_tokens, None
        if corr_tokens is None or corr_tokens.numel() == 0:
            return human_tokens, None

        batch_size, num_humans = human_tokens.shape[:2]
        corr_context = corr_tokens.mean(dim=1, keepdim=True).expand(-1, num_humans, -1)
        pose_context = pose_token.mean(dim=1, keepdim=True).expand(-1, num_humans, -1)
        context = torch.cat([human_tokens, corr_context, pose_context], dim=-1)
        hidden = self.context_mlp(context)
        delta_raw = self.delta_head(hidden)
        if self.max_delta is not None and self.max_delta > 0:
            delta = self.max_delta * torch.tanh(delta_raw)
        else:
            delta = delta_raw

        if self.use_gate:
            learned_gate = torch.sigmoid(self.gate_head(hidden))
        else:
            learned_gate = torch.ones_like(delta[..., :1])
        gate = learned_gate
        shared_gate_used = None
        if shared_gate is not None:
            shared_gate_used = shared_gate.to(device=delta.device, dtype=delta.dtype).reshape(batch_size, -1, 1)
            if shared_gate_used.shape[1] == 1:
                shared_gate_used = shared_gate_used.expand(-1, num_humans, -1)
            else:
                shared_gate_used = shared_gate_used[:, :num_humans]
            if self.gate_mode == "shared":
                gate = shared_gate_used
            elif self.gate_mode == "product":
                gate = shared_gate_used * learned_gate
        if apply_mask is not None:
            gate = gate * apply_mask.to(device=gate.device, dtype=gate.dtype)

        delta_applied = gate * delta
        corrected = human_tokens + delta_applied
        info = {
            "v8_human_latent_corr_delta_raw": delta_raw,
            "v8_human_latent_corr_delta_applied": delta_applied,
            "v8_human_latent_corr_gate": gate,
            "v8_human_latent_corr_learned_gate": learned_gate,
            "v8_human_latent_corr_delta_norm": delta_raw.norm(dim=-1),
        }
        if shared_gate_used is not None:
            info["v8_human_latent_corr_shared_gate"] = shared_gate_used
        return corrected, info
