import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorTokenProjector(nn.Module):
    """Project gathered ref/current patch evidence into decoder-dim AnchorTokens."""

    def __init__(self, enc_dim=1024, dec_dim=768, use_ref_feature=True):
        super().__init__()
        self.use_ref_feature = use_ref_feature
        geom_dim = 2 + 2 + 2 + 1
        in_dim = enc_dim + geom_dim + (enc_dim if use_ref_feature else 0)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dec_dim),
            nn.GELU(),
            nn.LayerNorm(dec_dim),
        )

    def forward(
        self,
        cur_feat,
        ref_feat,
        ref_pos_norm,
        cur_pos_norm,
        local_residual_norm,
        confidence,
        anchor_mask,
    ):
        confidence = confidence.clamp(0.0, 1.0).unsqueeze(-1)
        parts = [cur_feat]
        if self.use_ref_feature:
            parts.append(ref_feat)
        parts.extend([ref_pos_norm, cur_pos_norm, local_residual_norm, confidence])
        tokens = self.proj(torch.cat(parts, dim=-1))
        return tokens * anchor_mask.unsqueeze(-1).to(tokens.dtype)


class PoseAnchorAttention(nn.Module):
    """Cross-attention where only the pose token reads AnchorTokens."""

    def __init__(self, dec_dim=768, num_heads=8):
        super().__init__()
        assert dec_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dec_dim // num_heads
        self.q_proj = nn.Linear(dec_dim, dec_dim)
        self.k_proj = nn.Linear(dec_dim, dec_dim)
        self.v_proj = nn.Linear(dec_dim, dec_dim)
        self.out_proj = nn.Linear(dec_dim, dec_dim)
        self.norm_q = nn.LayerNorm(dec_dim)
        self.norm_kv = nn.LayerNorm(dec_dim)

    def forward(self, pose_token, anchor_tokens, anchor_mask):
        bsz, num_anchor, dim = anchor_tokens.shape
        valid = anchor_mask.bool()
        has_anchor = valid.any(dim=1)
        safe_valid = valid.clone()
        if num_anchor > 0:
            safe_valid[~has_anchor, 0] = True

        q = self.q_proj(self.norm_q(pose_token))
        k = self.k_proj(self.norm_kv(anchor_tokens))
        v = self.v_proj(self.norm_kv(anchor_tokens))

        q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, num_anchor, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, num_anchor, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~safe_valid[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = attn * has_anchor[:, None, None, None].to(attn.dtype)

        context = attn @ v
        context = context.transpose(1, 2).reshape(bsz, 1, dim)
        context = self.out_proj(context)
        context = context * has_anchor[:, None, None].to(context.dtype)
        return context, attn.detach()


class AnchorPoseAdapter(nn.Module):
    """Decoder-after V6-A adapter that only applies bounded camera pose residuals."""

    def __init__(
        self,
        enc_dim=1024,
        dec_dim=768,
        num_heads=8,
        use_ref_feature=True,
        max_delta_t=0.25,
        max_delta_q=0.05,
    ):
        super().__init__()
        self.token_projector = AnchorTokenProjector(
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            use_ref_feature=use_ref_feature,
        )
        self.pose_anchor_attention = PoseAnchorAttention(dec_dim=dec_dim, num_heads=num_heads)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dec_dim * 3),
            nn.Linear(dec_dim * 3, dec_dim),
            nn.GELU(),
            nn.Linear(dec_dim, 7),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.max_delta_t = float(max_delta_t)
        self.max_delta_q = float(max_delta_q)

    def forward(
        self,
        pose_token,
        pose_base,
        ref_feat,
        cur_feat,
        ref_pos_norm,
        cur_pos_norm,
        local_residual_norm,
        confidence,
        quality_gate,
        anchor_mask,
        apply_rotation=False,
    ):
        anchor_mask = anchor_mask.bool()
        has_anchor = anchor_mask.any(dim=1, keepdim=True).to(pose_base.dtype)
        quality_gate = quality_gate.to(pose_base.dtype).clamp(0.0, 1.0).view(-1, 1)
        gate = quality_gate * has_anchor

        anchor_tokens = self.token_projector(
            cur_feat=cur_feat,
            ref_feat=ref_feat,
            ref_pos_norm=ref_pos_norm.to(cur_feat.dtype),
            cur_pos_norm=cur_pos_norm.to(cur_feat.dtype),
            local_residual_norm=local_residual_norm.to(cur_feat.dtype),
            confidence=confidence.to(cur_feat.dtype),
            anchor_mask=anchor_mask,
        )
        anchor_context, attn = self.pose_anchor_attention(pose_token, anchor_tokens, anchor_mask)
        delta_raw = self.delta_head(
            torch.cat([pose_token, anchor_context, pose_token * anchor_context], dim=-1)
        ).squeeze(1)

        pose_out = pose_base.clone()
        delta_t = torch.tanh(delta_raw[:, :3]) * self.max_delta_t * gate
        pose_out[:, :3] = pose_base[:, :3] + delta_t

        if apply_rotation:
            delta_q = torch.tanh(delta_raw[:, 3:7]) * self.max_delta_q * gate
            pose_out[:, 3:7] = F.normalize(pose_base[:, 3:7] + delta_q, p=2, dim=-1)
        else:
            delta_q = torch.zeros_like(delta_raw[:, 3:7])

        info = {
            "anchor_pose_gate": gate.squeeze(-1).detach(),
            "anchor_pose_delta_t_norm": delta_t.detach().norm(dim=-1),
            "anchor_pose_delta_q_norm": delta_q.detach().norm(dim=-1),
            "anchor_pose_attn_max": attn.amax(dim=(-1, -2, -3)),
            "anchor_pose_valid": has_anchor.squeeze(-1).detach(),
        }
        return pose_out, info
