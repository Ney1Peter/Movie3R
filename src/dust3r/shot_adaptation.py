# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

"""
Shot-Aware Adaptation Modules

包含两个轻量模块用于处理镜头跳变：
1. ShotTokenGenerator: 生成 shot token q_t
2. LoRA Layers: 对 base model 输出做 LoRA 风格微调修正
   - PoseLoRALayer: 对相机位姿做微调修正
   - HumanLoRALayer: 对 SMPL 人体参数做微调修正
   - WorldLoRALayer: 对场景点云做全局平移修正

设计原则：使用标准 LoRA 低秩分解形式 (W' = W + BA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShotTokenGenerator(nn.Module):
    """
    Shot Token Generator - 使用相邻帧差异生成 shot token

    基于全局特征差异：
    输入: F_dec[i] 和 F_dec[i-1]（decoder 输入的图像 token）
    输出: q_t [B, 1, dec_dim]

    q_t 编码了两帧之间的"不连续程度"，供后续 LoRA 层使用。
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


# **========== 原始代码 (Residual Adapter) ==========**

class StateGate(nn.Module):
    """
    状态门控模块（已移除）
    """
    def __init__(self, dec_dim=768):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dec_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, q_t):
        alpha = torch.sigmoid(self.gate_mlp(q_t))
        return alpha


class PoseResidualAdapter(nn.Module):
    """
    Pose Residual Adapter - 对相机位姿做微调修正（已改为 LoRA）

    pose_final = pose_base + gamma * delta_pose
    """
    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.adapter = nn.Sequential(
            nn.Linear(dec_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 7),
        )

    def forward(self, z_token, q_out, pose_base):
        x = torch.cat([z_token, q_out], dim=-1)
        delta = self.adapter(x).squeeze(1)
        t_base = pose_base[:, :3]
        q_base = pose_base[:, 3:7]
        delta_t = delta[:, :3]
        delta_q = delta[:, 3:7]
        t_final = t_base + self.gamma * delta_t
        q_final = F.normalize(q_base + self.gamma * delta_q, dim=-1)
        return torch.cat([t_final, q_final], dim=-1)


class HumanResidualAdapter(nn.Module):
    """
    Human Residual Adapter - 对 SMPL 人体参数做微调修正（已改为 LoRA）
    """
    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma_shape = nn.Parameter(torch.tensor(0.0))
        self.gamma_transl = nn.Parameter(torch.tensor(0.0))
        in_dim = dec_dim * 2
        self.adapter_shape = nn.Linear(in_dim, 10)
        self.adapter_transl = nn.Linear(in_dim, 3)

    def forward(self, smpl_token, q_out, pred_smpl_dict):
        N = smpl_token.shape[1]
        q_expand = q_out.expand(-1, N, -1)
        x = torch.cat([smpl_token, q_expand], dim=-1)
        out = pred_smpl_dict.copy()
        out['smpl_shape'] = pred_smpl_dict['smpl_shape'] + self.gamma_shape * self.adapter_shape(x)
        out['smpl_transl'] = pred_smpl_dict['smpl_transl'] + self.gamma_transl * self.adapter_transl(x)
        return out


class WorldResidualAdapter(nn.Module):
    """
    World Residual Adapter - 对场景点云做全局平移修正（已改为 LoRA）
    """
    def __init__(self, dec_dim=768):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        in_dim = dec_dim * 2
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, img_tokens, pose_token, q_out, world_base):
        img_global = img_tokens.mean(dim=1, keepdim=True)
        x = torch.cat([img_global, q_out], dim=-1)
        delta = self.adapter(x)
        delta = delta.squeeze(1).unsqueeze(1).unsqueeze(1)
        return world_base + self.gamma * delta

# **========== 结束 ==========**


# **========== 新代码 (LoRA) ==========**

class PoseLoRALayer(nn.Module):
    """
    Pose LoRA Layer - 对相机位姿做 LoRA 微调修正

    标准 LoRA 形式: pose_final = pose_base + gamma * delta
    其中 delta = B @ A(x)，A: input→rank, B: rank→output

    输入 (作为 condition/input):
        z_token: [B, 1, dec_dim] - decoder 输出的 refined pose token (z')
        q_out: [B, 1, dec_dim] - decoder 输出的 refined shot token (q')
        pose_base: [B, 7] - base model 输出的 trans(3) + quat(4)

    输出:
        pose_final: [B, 7] - 修正后的 trans(3) + quat(4)
    """

    def __init__(self, dec_dim=768, rank=64):
        super().__init__()
        self.rank = rank
        # 初始为 0，确保初始状态 final = base，不破坏 frozen base model
        self.gamma = nn.Parameter(torch.tensor(0.0))
        # LoRA: A (input→rank) + B (rank→output)
        # 输入是 concat(z_token, q_out)，维度是 2*dec_dim
        self.lora_A = nn.Linear(dec_dim * 2, rank, bias=False)  # 1536→64
        self.lora_B = nn.Linear(rank, 7, bias=False)              # 64→7

    def forward(self, z_token, q_out, pose_base):
        """
        Args:
            z_token: [B, 1, dec_dim] - refined pose token (condition)
            q_out: [B, 1, dec_dim] - refined shot token (condition)
            pose_base: [B, 7] - base prediction
        Returns:
            pose_final: [B, 7]
        """
        x = torch.cat([z_token, q_out], dim=-1)  # [B, 1, 2*dec_dim]
        # LoRA: delta = B @ A(x)
        delta = self.lora_B(self.lora_A(x)).squeeze(1)  # [B, 7]

        t_base = pose_base[:, :3]      # [B, 3]
        q_base = pose_base[:, 3:7]    # [B, 4]
        delta_t = delta[:, :3]       # [B, 3]
        delta_q = delta[:, 3:7]      # [B, 4]

        t_final = t_base + self.gamma * delta_t
        q_final = F.normalize(q_base + self.gamma * delta_q, dim=-1)

        return torch.cat([t_final, q_final], dim=-1)


class HumanLoRALayer(nn.Module):
    """
    Human LoRA Layer - 对 SMPL 人体参数做 LoRA 微调修正

    标准 LoRA 形式: y_final = y_base + gamma * delta
    其中 delta = B @ A(x)

    当前版本只修正 smpl_shape 和 smpl_transl，不修正 smpl_rotmat。

    输入 (作为 condition/input):
        smpl_token: [B, N_humans, dec_dim] - decoder 输出的人体 token (H')
        q_out: [B, 1, dec_dim] - decoder 输出的 refined shot token (q')
        pred_smpl_dict: dict with keys smpl_shape, smpl_transl, smpl_rotmat, smpl_expression

    输出:
        smpl_dict_final: dict (同输入结构)
    """

    def __init__(self, dec_dim=768, rank=64):
        super().__init__()
        self.rank = rank
        # 初始为 0，确保初始状态 final = base，不破坏 frozen base model
        self.gamma_shape = nn.Parameter(torch.tensor(0.0))
        self.gamma_transl = nn.Parameter(torch.tensor(0.0))

        in_dim = dec_dim * 2  # smpl_token + q_out

        # LoRA for smpl_shape (10D)
        self.lora_A_shape = nn.Linear(in_dim, rank, bias=False)
        self.lora_B_shape = nn.Linear(rank, 10, bias=False)

        # LoRA for smpl_transl (3D)
        self.lora_A_transl = nn.Linear(in_dim, rank, bias=False)
        self.lora_B_transl = nn.Linear(rank, 3, bias=False)

    def forward(self, smpl_token, q_out, pred_smpl_dict):
        """
        Args:
            smpl_token: [B, N_humans, dec_dim] - human token (condition)
            q_out: [B, 1, dec_dim] - refined shot token (condition)
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

        # LoRA: delta = B @ A(x)
        delta_shape = self.lora_B_shape(self.lora_A_shape(x))  # [B, N, 10]
        delta_transl = self.lora_B_transl(self.lora_A_transl(x))  # [B, N, 3]

        out['smpl_shape'] = pred_smpl_dict['smpl_shape'] + self.gamma_shape * delta_shape
        out['smpl_transl'] = pred_smpl_dict['smpl_transl'] + self.gamma_transl * delta_transl

        # rotmat / expression 保持不变
        return out


class WorldLoRALayer(nn.Module):
    """
    World LoRA Layer (Global Shift) - 对场景点云做全局平移修正

    标准 LoRA 形式: world_final = world_base + gamma * delta_world
    其中 delta_world = B @ A(x)

    注意：当前实现本质上是给整张 pointmap 加一个全局 3D 平移 residual，
    能修全局 world alignment / camera offset，但不能修局部几何。

    输入 (作为 condition/input):
        img_tokens: [B, N, dec_dim] - decoder 输出的图像 token (F')，内部做全局平均池化
        pose_token: [B, 1, dec_dim] - decoder 输出的 refined pose token (z')
        q_out: [B, 1, dec_dim] - decoder 输出的 refined shot token (q')
        world_base: [B, H, W, 3] - base model 输出的 pts3d

    输出:
        world_final: [B, H, W, 3]
    """

    def __init__(self, dec_dim=768, rank=64):
        super().__init__()
        self.rank = rank
        # 初始为 0，确保初始状态 final = base，不破坏 frozen base model
        self.gamma = nn.Parameter(torch.tensor(0.0))

        in_dim = dec_dim * 2  # pooled img feat + q_out

        # LoRA: A (input→rank) + B (rank→output)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)  # 1536→64
        self.lora_B = nn.Linear(rank, 3, bias=False)       # 64→3

    def forward(self, img_tokens, pose_token, q_out, world_base):
        """
        Args:
            img_tokens: [B, N, dec_dim] - decoder 输出的图像 token (condition)
            pose_token: [B, 1, dec_dim] - refined pose token (condition)
            q_out: [B, 1, dec_dim] - refined shot token (condition)
            world_base: [B, H, W, 3] - base prediction
        Returns:
            world_final: [B, H, W, 3]
        """
        # 全局平均池化（内部做），失去空间信息，只修全局偏移
        img_global = img_tokens.mean(dim=1, keepdim=True)  # [B, 1, dec_dim]

        x = torch.cat([img_global, q_out], dim=-1)  # [B, 1, 2*dec_dim]

        # LoRA: delta = B @ A(x)
        delta = self.lora_B(self.lora_A(x))  # [B, 1, 3]
        delta = delta.squeeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]

        return world_base + self.gamma * delta

# **========== 结束 ==========**
