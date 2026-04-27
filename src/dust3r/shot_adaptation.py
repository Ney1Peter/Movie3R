# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

"""
Shot-Aware Adaptation Modules

包含三个轻量模块用于处理镜头跳变：
1. ShotTokenGenerator: 生成 shot token q_t
2. StateGate: 调制 state 的更新
3. LoRA Heads: 对输出做微调修正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShotTokenGenerator(nn.Module):
    """
    Global Difference Token - 使用相邻帧差异生成 shot token

    V1: 基于全局特征差异
    输入: F_dec[i] 和 F_dec[i-1]（decoder 输入的图像 token）
    输出: q_t [B, 1, dec_dim]

    q_t 编码了两帧之间的"不连续程度"，供后续 StateGate 和 LoRA 使用。
    """

    def __init__(self, dec_dim=768):
        super().__init__()
        # V1: g_curr, g_prev, diff, sim = 3 * dec_dim + 1
        self.shot_mlp = nn.Sequential(
            nn.Linear(dec_dim * 3 + 1, 256),
            nn.GELU(),
            nn.Linear(256, dec_dim),
        )
        # i=0 没有 previous frame，用可学习的 q_init
        self.q_init = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    def forward(self, feat_curr, feat_prev, i):
        """
        Args:
            feat_curr: [B, N, dec_dim] 当前帧 decoder 输入特征
            feat_prev: [B, N, dec_dim] 上一帧 decoder 输入特征
            i: int 帧索引，i=0 时使用 q_init
        Returns:
            q_t: [B, 1, dec_dim]
        """
        if i == 0:
            return self.q_init.expand(feat_curr.shape[0], -1, -1)

        # 全局特征：mean pooling
        g_curr = feat_curr.mean(dim=1)      # [B, dec_dim]
        g_prev = feat_prev.mean(dim=1)      # [B, dec_dim]

        # 差异特征
        diff = g_curr - g_prev              # [B, dec_dim]

        # 相似度（余弦）
        sim = F.cosine_similarity(g_curr, g_prev, dim=-1)  # [B]

        # 拼接: [g_curr, g_prev, diff, sim]
        x = torch.cat([g_curr, g_prev, diff, sim.unsqueeze(-1)], dim=-1)  # [B, 3*dec_dim+1]

        # 生成 shot token
        q_t = self.shot_mlp(x).unsqueeze(1)  # [B, 1, dec_dim]

        return q_t


class StateGate(nn.Module):
    """
    状态门控模块

    S_tilde = alpha * S_prev + (1 - alpha) * S0

    alpha 由 shot token q_t 生成，控制状态保留程度：
    - alpha 接近 1：保留大部分旧状态（相机连续运动）
    - alpha 接近 0：重置为初始状态（镜头跳变后）
    """

    def __init__(self, dec_dim=768):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dec_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, q_t):
        """
        Args:
            q_t: [B, 1, dec_dim] shot token
        Returns:
            alpha: [B, 1, 1] 门控值 (0~1)
        """
        # alpha: [B, 1, 1]
        alpha = torch.sigmoid(self.gate_mlp(q_t))
        return alpha


class LoRAPoseHead(nn.Module):
    """
    Pose LoRA - 对相机位姿做微调修正

    y_final = y_base + gamma * delta_y

    输入:
        z_token: [B, 1, dec_dim] - decoder 输出的 pose token (z')
        q_out: [B, 1, dec_dim] - decoder 输出的 shot token (q')
        pose_base: [B, 7] - trans(3) + quat(4)

    输出:
        pose_final: [B, 7] - trans(3) + quat(4)
    """

    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.01))  # 工程建议值：0.01
        self.lora = nn.Sequential(
            nn.Linear(dec_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 7),  # delta_trans(3) + delta_quat(4)
        )

    def forward(self, z_token, q_out, pose_base):
        """
        Args:
            z_token: [B, 1, dec_dim]
            q_out: [B, 1, dec_dim]
            pose_base: [B, 7] - trans(3) + quat(4)
        Returns:
            pose_final: [B, 7]
        """
        x = torch.cat([z_token, q_out], dim=-1)  # [B, 1, 2*dec_dim]
        delta = self.lora(x).squeeze(1)  # [B, 7]

        t_base = pose_base[:, :3]      # [B, 3]
        q_base = pose_base[:, 3:7]    # [B, 4]
        delta_t = delta[:, :3]       # [B, 3]
        delta_q = delta[:, 3:7]      # [B, 4]

        t_final = t_base + self.gamma * delta_t
        q_final = F.normalize(q_base + self.gamma * delta_q, dim=-1)

        return torch.cat([t_final, q_final], dim=-1)


class LoRAHumanHead(nn.Module):
    """
    Human LoRA - 对 SMPL 人体参数做微调修正（第一版）

    y_final = y_base + gamma * delta_y

    第一版只修正 smpl_shape 和 smpl_transl，不修正 smpl_rotmat（rotation matrix
    直接相加后不再是合法旋转矩阵，需要用 axis-angle / 6D rotation residual）

    输入:
        smpl_token: [B, N_humans, dec_dim] - decoder 输出的人体 token (H')
        q_out: [B, 1, dec_dim] - decoder 输出的 shot token (q')
        pred_smpl_dict: dict with keys smpl_shape, smpl_transl, smpl_rotmat, smpl_expression

    输出:
        smpl_dict_final: dict (同输入结构)
    """

    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma_shape = nn.Parameter(torch.tensor(0.01))  # 工程建议值：0.01
        self.gamma_transl = nn.Parameter(torch.tensor(0.01))  # 工程建议值：0.01

        in_dim = dec_dim * 2  # smpl_token + q_out

        # smpl_shape: betas 10D
        self.lora_shape = nn.Linear(in_dim, 10)
        # smpl_transl: 3D
        self.lora_transl = nn.Linear(in_dim, 3)
        # 注意：第一版不修 rotmat（rotation matrix 直接相加不再是合法旋转矩阵）

    def forward(self, smpl_token, q_out, pred_smpl_dict):
        """
        Args:
            smpl_token: [B, N_humans, dec_dim]
            q_out: [B, 1, dec_dim]
            pred_smpl_dict: dict with smpl_shape(B,N,10), smpl_transl(B,N,3),
                          smpl_rotmat(B,N,6,3,3), smpl_expression(B,N,10)
        Returns:
            smpl_dict_final: dict
        """
        N = smpl_token.shape[1]
        q_expand = q_out.expand(-1, N, -1)  # [B, N, dec_dim]
        x = torch.cat([smpl_token, q_expand], dim=-1)  # [B, N, 2*dec_dim]

        # 不 inplace 修改原 dict
        out = pred_smpl_dict.copy()

        out['smpl_shape'] = pred_smpl_dict['smpl_shape'] + self.gamma_shape * self.lora_shape(x)

        out['smpl_transl'] = pred_smpl_dict['smpl_transl'] + self.gamma_transl * self.lora_transl(x)

        # rotmat / expression 保持不变
        return out


class LoRAWorldGlobalShift(nn.Module):
    """
    World LoRA (Global Shift) - 对场景点云做全局平移修正

    y_final = y_base + gamma * delta_y

    注意：当前实现本质上是给整张 pointmap 加一个全局 3D 平移 residual，
    能修全局 world alignment / camera offset，但不能修局部几何。

    输入:
        img_tokens: [B, N, dec_dim] - decoder 输出的图像 token (F')，内部做全局平均池化
        pose_token: [B, 1, dec_dim] - decoder 输出的 pose token (z')
        q_out: [B, 1, dec_dim] - decoder 输出的 shot token (q')
        world_base: [B, H, W, 3] - DPT 输出的 pts3d

    输出:
        world_final: [B, H, W, 3]
    """

    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.01))  # 工程建议值：0.01

        in_dim = dec_dim * 2  # pooled img feat + q_out
        self.lora = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # delta xyz
        )

    def forward(self, img_tokens, pose_token, q_out, world_base):
        """
        Args:
            img_tokens: [B, N, dec_dim] - decoder 输出的图像 token，内部做全局平均池化
            pose_token: [B, 1, dec_dim] - z'
            q_out: [B, 1, dec_dim] - q'
            world_base: [B, H, W, 3] - pts3d
        Returns:
            world_final: [B, H, W, 3]
        """
        # 全局平均池化（内部做），失去空间信息，只修全局偏移
        img_global = img_tokens.mean(dim=1, keepdim=True)  # [B, 1, dec_dim]

        x = torch.cat([img_global, q_out], dim=-1)  # [B, 1, 2*dec_dim]
        delta = self.lora(x)  # [B, 1, 3]
        delta = delta.squeeze(1)  # [B, 3]
        delta = delta.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 3]

        return world_base + self.gamma * delta
