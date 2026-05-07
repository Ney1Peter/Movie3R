# Movie3R 模型设计文档

> **旧版内容备份**：本文原始内容记录的是 StateGate + Residual Adapter 设计，其中 StateGate、Residual Adapter、强制/门控 state reset 等描述已经过时。当前实现已更新为 ShotTokenGenerator + LoRA Head V1，并保持原 Human3R recurrent state 行为。新设计说明将在文档前部新增，旧内容保留用于回溯。

## 当前实现（2026/05/07）

### 0.1 设计目标

当前 Movie3R V1 目标不是重新训练 CUT3R/Human3R 主体，而是在冻结原模型参数的前提下，通过新增 shot token 和 LoRA-style 低秩修正模块，对镜头跳变导致的位置、朝向和全局对齐问题做小范围修正。

重点修正：
- 相机位置和朝向
- 场景/背景 pointmap 的全局位置偏移
- 人体整体平移

暂不修正：
- 人体 shape
- 人体 body pose 细节
- 表情 expression
- pointmap 局部几何细节

### 0.2 与原 Human3R 的关系

原 Human3R 主路径保持不变：

```text
image
  -> CUT3R encoder image tokens
  -> MHMR/DINOv2 检测人体位置
  -> 提取 MHMR token + CUT3R token
  -> mlp_fuse 得到 human token
  -> [pose, image, human] 进入 recurrent decoder
  -> 输出 camera pose / pointmap / SMPL 参数
```

Movie3R V1 只在该路径上增加 shot token：

```text
F_i, F_{i-1}
  -> ShotTokenGenerator
  -> q_i
  -> [pose, image, human, q_i] 进入 recurrent decoder
  -> [z', F', H', q'_i]
  -> LoRA Head V1 修正最终输出
```

### 0.3 State 行为

当前 V1 不使用 StateGate，也不使用 shot token 控制 state reset。

训练路径默认保持原 Human3R 行为：

```python
state_for_recurrent = state_feat
```

也就是说，decoder 仍参考前一帧 recurrent state。Shot token 只作为额外 prompt 参与 decoder cross-attention，不改变 state 更新策略。

### 0.4 ShotTokenGenerator

Shot token 使用 decoder 输入 image token 生成：

```python
g_curr = feat_curr.mean(dim=1)
g_prev = feat_prev.mean(dim=1)
diff = g_curr - g_prev
sim = cosine_similarity(g_curr, g_prev)
q_t = MLP([g_curr, g_prev, diff, sim])
```

`q_t` 进入 decoder 前是输入 shot token，decoder 输出后的最后一个 token 是 `q'_t`。LoRA heads 使用 `q'_t` 作为 condition。

### 0.5 LoRA Head V1 修正范围

| 模块 | 输入 condition | 修正输出 | 不修正 |
|------|-----------------|----------|--------|
| PoseLoRALayer | `z' + q'` | `camera_pose` translation + quaternion | - |
| HumanLoRALayer | `H' + q'` | `smpl_transl` | `smpl_shape` / `smpl_rotmat` / `smpl_expression` |
| WorldLoRALayer | pooled `F' + q'` | `pts3d_in_self_view` / `pts3d_in_other_view` 全局 3D shift | 局部 pointmap 几何 |

当前 LoRA 是新增低秩修正模块，形式为：

```python
delta = lora_B(lora_A(condition))
output = base_output + gamma * delta
```

原 Human3R/CUT3R 参数被冻结，不被 LoRA 直接修改。

### 0.6 可训练参数范围

`freeze='shot_adaptation'` 下训练：

| 模块 | 当前作用 | rank=128 估算参数量 |
|------|----------|--------------------|
| ShotTokenGenerator | 生成 shot token | ~788K |
| PoseLoRALayer | 修正相机位姿 | ~198K |
| HumanLoRALayer | 修正人体平移 | ~197K |
| WorldLoRALayer | 修正 pointmap 全局平移 | ~197K |
| **总计** | | **~1.38M** |

冻结：encoder、decoder、DINOv2/MHMR backbone、downstream heads、原始 human heads、原始 state tokens。

### 0.7 shot_label 使用策略

当前 V1 暂不使用 `shot_label` 作为显式监督。

当前判断方式是隐式的：

```text
相邻帧 image token 差异 -> q_t -> decoder -> q'_t -> LoRA 修正 -> task loss 反向传播
```

也就是说，模型通过最终 camera/world/human task loss 学习何时需要更大修正，而不是直接训练一个 shot-change classifier。

后续 V2 可选方案：
- 给 ShotTokenGenerator 增加 `shot_logit`
- 使用数据集中的 `shot_label` 做 BCE 辅助监督
- 仅作为辅助 loss，不直接硬开关 LoRA
- 如重新引入 StateGate，再用 shot probability 控制 state mixing

### 0.8 当前限制

- Inference 路径 `forward_recurrent_lighter` 已接入 ShotToken/LoRA，但 LoRA64 正式训练权重推理失败
- LoRA rank 已进入 config，当前正式训练使用 `lora_rank=64`
- 当前 `q_t` 直接 append 到 decoder token 序列，不是 residual-safe；即使 LoRA gamma=0，额外 token 也会通过 decoder attention 改变 base 输出
- 当前 loss/val 指标不能充分代表 demo 可视化质量，缺少 camera/pointmap/SMPL 的绝对尺度监控
- 旧 LoRA checkpoint 与 LoRA Head V1 的 HumanLoRA 结构不完全兼容，不建议继续使用

### 0.9 LoRA64 推理失败诊断（2026/05/07）

LoRA64 正式训练完成后，`checkpoint-best.pth` 在 `data/h36.mp4` 上出现严重视觉错误：相机尺度异常、人体过小、场景点云和人体乱成一团。

同一段视频前 8 帧的消融结果：

| 模式 | camera 平移均值 | pointmap 范围均值 | SMPL 平移均值 | 结论 |
|------|----------------|------------------|---------------|------|
| base Human3R | `0.010` | `9.049` | `4.935` | 原模型尺度正常 |
| LoRA64 checkpoint，关闭 `enable_shot_adaptation` | `0.010` | `9.049` | `4.935` | checkpoint 中冻结 base 权重正常 |
| LoRA64 checkpoint，打开 `enable_shot_adaptation` | `0.042` | `3.844` | `3.067` | shot adaptation 分支破坏尺度 |
| LoRA64 checkpoint，LoRA gamma 全置 0，仅保留 trained shot token | `0.020` | `3.844` | `1.966` | pointmap 崩坏主要来自 trained shot token 进入 decoder |

当前判断：
- checkpoint 加载正常，base 权重没有被破坏
- `enable_shot_adaptation=False` 只是推理时跳过新增 shot/LoRA 分支，因为 base 在训练中被冻结，所以输出恢复到原 Human3R 水平
- LoRA residual 不是唯一问题，trained shot token 本身进入 frozen decoder 后就足以改变输出尺度

下一步需要先验证 shot token 质量，而不是继续训练：
- 检查 `AvatarReX_Video` / `AvatarReX_AABB` 的 `shot_label`
- 统计 `g_curr/g_prev` 的 cosine similarity 和 diff norm 是否区分连续/跳变
- 统计训练前/训练后 `q_t` 的范数、cosine、聚类和二分类可分性
- 增加输出尺度诊断指标，确认 `q_t` 注入不会无约束破坏 base 输出

### 0.10 Shot Token 质量验证补充（2026/05/07）

诊断脚本：`scripts/analyze_shot_token.py`

数据集格式检查通过：

| 数据集 | 期望 `shot_label` | 实测 | 结论 |
|------|------------------|------|------|
| `AvatarReX_Video` | `[0, 0, 0, 0]` | 全部符合 | 数据格式正确 |
| `AvatarReX_AABB` | `[0, 0, 1, 0]` | 全部符合 | 数据格式正确 |

`ShotTokenGenerator` 的输入特征 `g_curr/g_prev` 本身具备很强区分度：

| 指标 | 连续帧 | 跳变帧 | AUC |
|------|--------|--------|-----|
| `cosine(g_curr, g_prev)` | `0.9990` | `0.9889` | `0.9997` |
| `||g_curr - g_prev||` | `0.697` | `2.445` | `0.9997` |

虽然 `0.9990` 和 `0.9889` 都接近 1，但在高维 token 特征中这个差距已经足够稳定；同时 `diff_norm` 跳变帧约为连续帧的 3.5 倍，AUC 接近 1，说明输入信号不是瓶颈。

训练后的 `q_t` 存在尺度和 no-op 问题：

| 指标 | 连续帧 | 跳变帧 | 结论 |
|------|--------|--------|------|
| `||q_t||` | `62.17` | `62.21` | 范数不区分跳变，且过大 |
| view2 `||q_t - q_{t-1}||` | `1.21` | `5.22` | 有跳变响应 |
| decoder image token norm | `23.15` | - | `q_t` 约为普通 token 的 2.75 倍 |

当前判断：`q_t` 不是完全没学到跳变，而是学成了一个始终很强的全局 prompt。连续帧也会强干预 decoder，没有学到 no-op 行为。

这里的 no-op 指连续帧时 `q_t` 应尽量“不产生操作”：不明显改变原 Human3R 的 camera、pointmap 和 SMPL 输出；只有在 `shot_label=1` 的跳变帧才允许更强干预。

### 0.11 下一版 Shot Token 约束方案

推荐保留 shot token 进入 decoder 的设计，但必须让它可控：

```text
g_curr, g_prev, diff, sim
    -> ShotTokenGenerator
    -> q_raw, shot_logit
    -> shot_prob = sigmoid(shot_logit)
    -> q_t = shot_scale * shot_prob * LayerNorm(q_raw)
    -> [pose, image, human, q_t]
```

建议新增约束：

| 约束 | 目的 | 建议优先级 |
|------|------|-----------|
| `shot_logit + BCE(shot_label)` | 让模块显式知道哪里是跳变 | 最高 |
| `shot_prob` gate | 连续帧自动弱化 `q_t`，跳变帧增强 `q_t` | 最高 |
| `LayerNorm(q_raw)` | 控制 `q_t` 尺度，不再比 decoder token 大数倍 | 高 |
| `shot_scale` 初始接近 0 | 保证训练初期接近原 Human3R，避免一开始破坏 base | 高 |
| 连续帧 no-op loss | 约束 `shot_label=0` 时输出接近 shot-off/base 输出 | 高 |
| `q_norm` / 输出尺度监控 | 防止再次出现 loss 降但推理尺度崩 | 高 |

训练 loss 可加入：

```text
L = L_task
  + lambda_shot * BCE(shot_logit, shot_label)
  + lambda_q0 * (1 - shot_label) * ||q_t||^2
  + lambda_noop * (1 - shot_label) * ||pred_on - pred_off||
```

其中 `pred_off` 可以是关闭 shot token 的 frozen base 输出或 stop-gradient teacher 输出。第一版可先实现 `BCE + gate + LayerNorm + q_norm/no-op 监控`，再决定是否加入完整 `pred_on/pred_off` no-op loss。

当前执行计划改为三层都做：

1. Layer 1：新增 `shot_logit` 和 `shot_label` BCE 辅助监督。
2. Layer 2：新增 `shot_prob` gate、`LayerNorm(q_raw)` 和 `shot_scale`，控制 `q_t` 强度。
3. Layer 3：新增连续帧 no-op output distillation，直接约束 `shot_label=0` 时 `pred_with_shot` 接近 `pred_without_shot`。

为了方便回退，每层都拆成两个 commit：先注释备份旧代码，再新增实现。

---

## 1. 概述

### 1.1 任务背景

Movie3R 基于 Human3R (AvatarReX) 模型，针对**多镜头电影级人体重建**场景进行优化。核心挑战是处理镜头跳变（shot change）带来的时序不连续问题。

### 1.2 设计目标

- **快速适应镜头跳变**：当相机视角发生突变时，快速重置状态，避免错误累积
- **保持时序连续性**：在相机连续运动时，充分利用历史信息
- **轻量微调**：只训练 ~1.3M 新参数，不破坏预训练模型能力

---

## 2. 整体架构

### 2.1 模块组成

```
Shot-Aware Adaptation Modules
├── ShotTokenGenerator    (~787K)  生成 shot token q_t
├── StateGate             (~99K)   生成状态门控值 α
├── PoseResidualAdapter   (~198K)  修正相机位姿
├── HumanResidualAdapter  (~20K)   修正 SMPL 人体参数
└── WorldResidualAdapter  (~197K)  修正场景点云
────────────────────────────────────────────
总计 trainable: ~1.3M 参数
```

### 2.2 数据流

```
Frame i-1                     Frame i
    │                             │
    ▼                             ▼
encoder(feat_{i-1})    encoder(feat_i)
    │                             │
    ▼                             ▼
decoder_embed()          decoder_embed()
    │                             │
    │◄─────────────────────────────┤
    │        ShotTokenGenerator     │
    │           q_t = f(feat_i, feat_{i-1})
    │                             │
    ▼                             ▼
StateGate(q_t)             StateGate(q_t)
  α = sigmoid(MLP(q_t))    α = sigmoid(MLP(q_t))
    │                             │
    ▼                             ▼
S_t = α*S_prev + (1-α)*S0  S_t = α*S_prev + (1-α)*S0
    │                             │
    ├─────────────────────────────►┤
    │        Concat to f_img       │
    │   [pose, img, smpl, q_t]    │
    ▼                             ▼
              Decoder
    (Cross Attention: state ↔ all tokens)
    │
    ▼
  Output: [z_out, img_tokens, h_token, q_out]
    │
    ├─────────────────────────────►┤
    │        Residual Adapters      │
    │   PoseResidualAdapter        │
    │   HumanResidualAdapter       │
    │   WorldResidualAdapter       │
    ▼                             ▼
  Final Predictions
```

---

## 3. 各模块详解

### 3.1 ShotTokenGenerator

#### 3.1.1 功能
基于相邻帧的差异生成 shot token q_t，编码两帧之间的"不连续程度"。

#### 3.1.2 实现

```python
class ShotTokenGenerator(nn.Module):
    def forward(self, feat_curr, feat_prev, i):
        if i == 0:
            return self.q_init  # 可学习初始化

        # 全局特征：mean pooling
        g_curr = feat_curr.mean(dim=1)      # [B, dec_dim]
        g_prev = feat_prev.mean(dim=1)      # [B, dec_dim]

        # 差异特征
        diff = g_curr - g_prev              # [B, dec_dim]

        # 相似度（余弦）
        sim = F.cosine_similarity(g_curr, g_prev, dim=-1)  # [B]

        # 拼接并过 MLP
        x = torch.cat([g_curr, g_prev, diff, sim.unsqueeze(-1)], dim=-1)
        q_t = self.shot_mlp(x).unsqueeze(1)  # [B, 1, dec_dim]

        return q_t
```

#### 3.1.3 设计选择

| 选择 | 方案 | 理由 |
|------|------|------|
| 输入特征 | decoder 输入的 image token (f_dec) | 维度匹配 decoder，特征更精炼 |
| 聚合方式 | Mean pooling | 轻量快速，全局差异足以检测镜头跳变 |
| 特征组合 | [g_curr, g_prev, diff, sim] | 保留绝对信息和相对差异 |

#### 3.1.4 为什么用全局特征而非 patch-level？

- **全局特征足够**：镜头跳变通常是整体场景的变化（如切到不同机位），全局特征能捕捉
- **计算轻量**：避免 patch 级别的相似度矩阵计算
- **避免局部干扰**：背景小物体的移动不应触发跳变检测

未来如需更精细的跳变检测（如画面内物体突然变换），可考虑 patch-level 对比。

---

### 3.2 StateGate

#### 3.2.1 功能
根据 shot token q_t 生成门控值 α，控制状态 S 的更新程度。

#### 3.2.2 实现

```python
class StateGate(nn.Module):
    def forward(self, q_t):
        alpha = torch.sigmoid(self.gate_mlp(q_t))  # [B, 1, 1]
        return alpha  # ∈ [0, 1]
```

#### 3.2.3 状态更新机制

```
S_t = α * S_{t-1} + (1 - α) * S_0
```

| α 值 | 含义 | 场景 |
|------|------|------|
| α ≈ 1 | 保留大部分旧状态 | 相机连续运动 |
| α ≈ 0 | 重置为初始状态 | 镜头跳变后 |

#### 3.2.4 与 reset mask 的关系

StateGate 是**软性门控**，而 reset mask 是**硬性重置**：
- `reset=True`：跳过 StateGate，直接使用 S_0
- `reset=False` + `α ≈ 0`：通过 StateGate 重置

两者可以结合使用。

---

### 3.3 Decoder 中的 Shot Token 交互

#### 3.3.1 集成方式

Shot token q_t 在进入 decoder 前拼接到 f_img：

```python
# model.py _decoder()
f_img = torch.cat([f_img, f_shot], dim=1)  # [pose, img, smpl, q_t]
pos_img = torch.cat([pos_img, pos_shot], dim=1)
```

#### 3.3.2 Cross Attention 双向交互

Decoder Block 中的 cross attention 是双向的：

```
blk_state(x, y):
    Query = x = state
    Key/Value = y = [pose, img, smpl, q_t]
    → state attends to q_t (信息从 q_t 流向 state)

blk_img(y, x):
    Query = y = [pose, img, smpl, q_t]
    Key/Value = x = state
    → q_t attends to state (信息从 state 流向 q_t)
```

#### 3.3.3 Decoder 输出

Decoder 输出 `dec = [z', F', H', q']`，其中：
- `z'`：refined pose token
- `F'`：refined image tokens
- `H'`：refined human tokens
- `q'`：refined shot token（经过双向交互后的版本）

---

### 3.4 Residual Adapters

#### 3.4.1 为什么用 Residual Adapter？

**不是直接预测输出，而是预测修正量：**

```
pose_final = pose_base + γ * Δ_pose
```

**优势：**
1. **初始安全**：γ=0 时，`pose_final = pose_base`，完全保留预训练能力
2. **学习目标小**：只需学习"如何修正"，而非从头预测
3. **训练稳定**：不会产生远离预训练的输出

#### 3.4.2 标准 LoRA vs Residual Adapter

| 特征 | 标准 LoRA | Residual Adapter (当前实现) |
|------|-----------|---------------------------|
| 机制 | 低秩分解插入已有层 W → W + BA | 独立 MLP 预测 Δ |
| 复杂度 | 需修改原模型结构 | 直接添加，不触碰原模型 |
| 适用场景 | 适配已有权重 | 适配任意输出空间 |

当前实现更接近 **residual correction head**，虽叫 LoRA 但机制不同。

#### 3.4.3 输入：Condition / Input，非监督目标

Refined tokens (z_out, q_out) 是 **condition/input**，不是监督目标：
- 它们提供"在什么情况下需要修正"的上下文
- adapter 学习的是：给定这个上下文，如何修正 base prediction
- 监督信号来自最终 prediction 与 GT 的 loss

#### 3.4.4 PoseResidualAdapter

```python
class PoseResidualAdapter(nn.Module):
    def forward(self, z_token, q_out, pose_base):
        x = torch.cat([z_token, q_out], dim=-1)  # [B, 2*dec_dim]
        delta = self.adapter(x)  # 预测 [Δt, Δq]

        t_final = t_base + γ * Δt
        q_final = normalize(q_base + γ * Δq)

        return [t_final, q_final]
```

**输入 (condition)：**
- `z_token`：refined pose token
- `q_out`：refined shot token
- `pose_base`：base model 输出的位姿

**输出：**
- 修正后的 trans(3) + quat(4)

**注意**：对于旋转部分，直接相加后 normalize 只是简化处理。更严谨的做法是使用 axis-angle 或李代数。

#### 3.4.5 HumanResidualAdapter

```python
class HumanResidualAdapter(nn.Module):
    def forward(self, smpl_token, q_out, pred_smpl_dict):
        # 修正 smpl_shape 和 smpl_transl
        out['smpl_shape'] = base + γ_shape * adapter_shape(x)
        out['smpl_transl'] = base + γ_transl * adapter_transl(x)
        # smpl_rotmat 保持不变（避免破坏合法旋转矩阵）
```

**修正范围：**
- ✅ smpl_shape (10D)
- ✅ smpl_transl (3D)
- ❌ smpl_rotmat (旋转矩阵直接相加不再是合法旋转)

#### 3.4.6 WorldResidualAdapter

```python
class WorldResidualAdapter(nn.Module):
    def forward(self, img_tokens, pose_token, q_out, world_base):
        img_global = img_tokens.mean(dim=1)  # 全局池化
        x = torch.cat([img_global, q_out], dim=-1)
        delta = self.adapter(x)  # [B, 1, 3]

        return world_base + γ * delta
```

**特性：**
- 全局平均池化，只修全局偏移
- 不修局部几何细节

---

## 4. 训练策略

### 4.1 Freeze 模式

```python
if freeze == 'shot_adaptation':
    # 冻结所有原始模块
    freeze_all_params(encoder)
    freeze_all_params(decoder)
    freeze_all_params(backbone)
    freeze_all_params(heads)

    # 只训练 shot adaptation 模块
    for module in [shot_token_generator, state_gate,
                   pose_residual_adapter, human_residual_adapter,
                   world_residual_adapter]:
        for p in module.parameters():
            p.requires_grad = True
```

### 4.2 参数量对比

| 模块 | 冻结状态 | 参数量 |
|------|----------|--------|
| ShotTokenGenerator | ✅ 训练 | ~787K |
| StateGate | ✅ 训练 | ~99K |
| PoseResidualAdapter | ✅ 训练 | ~198K |
| HumanResidualAdapter | ✅ 训练 | ~20K |
| WorldResidualAdapter | ✅ 训练 | ~197K |
| **新模块总计** | | **~1.3M** |
| Encoder (ViT) | ❌ 冻结 | ~600M |
| Decoder | ❌ 冻结 | ~226M |
| Backbone (Dinov2) | ❌ 冻结 | ~304M |
| Downstream Heads | ❌ 冻结 | ~152M |

**只训练 0.1% 的参数！**

### 4.3 Gamma 初始化

所有 γ 参数初始化为 **0.0**：
- 确保初始状态：`final = base`，不破坏预训练
- 随训练逐渐学习到合适的修正量

### 4.4 Loss 与监督

Residual adapter **没有独立的 loss**：
- 通过 final prediction 与 GT 的 task loss 端到端学习
- Loss 反向传播路径：
  ```
  GT → Task Loss → Final Prediction → Δ = adapter(condition)
  ```

---

## 5. 设计原理总结

### 5.1 为什么需要 Shot Token？

- 检测相邻帧的不连续程度
- 区分"相机连续运动"和"镜头跳变"
- 为后续修正提供上下文信息

### 5.2 为什么需要 StateGate？

- 镜头跳变后，旧 state 可能导致错误累积
- 通过 α 控制：保留多少旧状态，重置多少到 S_0
- 软性门控比硬性重置更平滑

### 5.3 为什么需要 Residual Adapter？

- 不破坏预训练模型的能力
- 只学习"修正量"，训练更稳定
- 模块化设计，可独立调整各部分

### 5.4 为什么 Shot Token 要参与 Decoder？

- 让 shot 信息影响所有 token 的 refinement
- q_t 通过 cross attention 与 pose/img/smpl token 交互
- 最终 q_out 是双向交互后的结果

---

## 6. 与原 Human3R 的区别

| 方面 | Human3R | Movie3R (Shot-Aware) |
|------|---------|----------------------|
| 状态更新 | 固定更新，无区分 | StateGate 软性门控 |
| 镜头跳变 | reset mask 硬性重置 | StateGate + reset 结合 |
| 输出修正 | 无 | Residual Adapter 微调 |
| 训练方式 | 全量微调 | 只训练 ~1.3M 参数 |
| 镜头跳变处理 | 依赖显式 reset 信号 | 自动检测 + 修正 |

---

## 7. 未来可能的改进方向

1. **Patch-level 对比**：更精细的跳变检测
2. **Rotation 表示**：使用 axis-angle 或李代数替代 quaternion 相加
3. **Local World Correction**：World adapter 增加空间感知能力
4. **SMPL Rotmat 修正**：学习合理的 rotation residual
5. **辅助 Loss**：加 Δ L2 / smoothness 正则，防止修正过大
