# Movie3R TODO List

## 训练代码方面

### 1. NativeScalerWithGradNormCount 与 Gradient Accumulation

#### 当前实现

**梯度累积由 `accelerator.accumulate(model)` 控制**：
```python
# train.py L512
for data_iter_step, batch in enumerate(data_loader):
    with accelerator.accumulate(model):  # ← 控制何时更新参数
        loss_scaler(loss, optimizer, parameters=model.parameters(), 
                    update_grad=True, clip_grad=1.0)
        optimizer.zero_grad()
```

**NativeScalerWithGradNormCount 只负责执行 backward + step**：
```python
# croco/utils/misc.py
class NativeScalerWithGradNormCount:
    def __call__(self, loss, optimizer, ..., update_grad=True):
        self.accelerator.backward(loss)  # 反向传播
        if update_grad:
            # 梯度裁剪 + optimizer.step()
            optimizer.step()
```

#### 两者关系

| 组件 | 职责 |
|------|------|
| `accelerator.accumulate(model)` | 控制"何时"调用 step（梯度累积） |
| `NativeScalerWithGradNormCount` | 执行 backward + step + 梯度裁剪 |

#### 结论
- ✅ **不冲突**，协同工作
- ✅ 当前实现合理，无需改动
- ✅ NativeScaler 只做 loss scaling（bf16）和梯度裁剪
- ✅ 梯度累积由 Accelerate 的 `accumulate()` 控制

#### 参数更新流程（重要）

```
train.py L512: with accelerator.accumulate(model):
                    │
                    ├── 决定是否允许更新
                    │   (accum_iter 控制)
                    ▼
                loss_scaler(loss, optimizer, ...)
                    │
                    ├── accelerator.backward(loss)
                    │       ↓
                    │   反向传播，梯度累加到 .grad
                    │
                    ├── accelerator.clip_grad_norm_(...)
                    │       ↓
                    │   梯度裁剪 (clip_grad=1.0)
                    │
                    └── optimizer.step()
                            ↓
                        AdamW 实际更新参数
                            ↓
                        optimizer.zero_grad()
                            ↓
                        清空 .grad，准备下一轮
```

**三方职责分工**：

| 组件 | 职责 | 谁负责 |
|------|------|--------|
| 控制"何时"更新 | 梯度累积逻辑 | `accelerator.accumulate(model)` |
| 执行反向传播 | bf16 scaling + backward | `NativeScaler` |
| 执行参数更新 | `optimizer.step()` | `AdamW (optimizer)` |
| 梯度裁剪 | `clip_grad_norm_` | `NativeScaler` |

> 注：虽然 NativeScaler 调用了 `optimizer.step()`，但 optimizer 本身是独立创建的（AdamW），NativeScaler 只是"调用者"。

---

### 2. 分布式训练框架设计

#### 当前配置
| 组件 | 使用 | 说明 |
|------|------|------|
| 多卡启动 | `torchrun` | PyTorch 原生分布式启动 |
| 分布式框架 | **Accelerate** | 封装 DDP，提供高层 API |
| 底层通信 | DDP (NCCL) | Accelerate 底层使用 |
| 混合精度 | bf16 | Accelerate 内置支持 |
| 梯度累积 | Accelerate 内置 | `gradient_accumulation_steps` |

#### 框架对比

| 框架 | 开发方 | 核心特性 | 适用场景 |
|------|--------|----------|----------|
| **DDP** | PyTorch | 原始分布式，数据并行 | 需要完全控制时 |
| **Accelerate** | HuggingFace | 封装 DDP，简洁 API | 推荐，当前使用 |
| **FSDP** | Meta | 分片模型参数到多卡 | 超大模型（百亿参数） |
| **DeepSpeed** | 微软 | ZeRO 优化器，分片状态 | 超大模型 |

#### 当前架构
```
torchrun --nproc_per_node=N
    │
    ▼
Accelerate(gradient_accumulation_steps, mixed_precision="bf16")
    │
    ▼
DistributedDataParallel (DDP) + NCCL
    │
    ▼
Model + Optimizer
```

#### 分析
- [TODO] 当前 DDP + Accelerate 架构是否合理？
  - ✅ 优点：配置简单，代码量少，H800 80GB 显存充足
  - ✅ 适合 ~1.3M 可训练参数的中小规模模型
  - ❌ 不适合超大规模模型（需要 FSDP/DeepSpeed）

#### 决策
- [TODO] 对于当前规模（1.3M 参数），DDP + Accelerate 足够
- [TODO] 如后续扩展到超大模型，再考虑 FSDP/DeepSpeed

---

### 3. 梯度累积原理与步数计算

#### 当前配置
```yaml
batch_size: 8
accum_iter: 1
epochs: 40
```

#### 关键参数
- `num_views: 4` — 每个 sample 包含 4 个视角
- `train_dataset: 800 @ 6 个 AvatarReX 数据集 = 4800 samples/epoch`

#### 步数计算

**每个 epoch 的步数（steps_per_epoch）**：
```
steps_per_epoch = 训练样本数 / (batch_size × num_gpus)

单卡训练：
  steps_per_epoch = 4800 / 8 = 600 steps/epoch

4卡训练：
  steps_per_epoch = 4800 / (8 × 4) = 150 steps/epoch

8卡训练：
  steps_per_epoch = 4800 / (8 × 8) = 75 steps/epoch
```

**当 accum_iter > 1 时的参数更新次数**：
```
单卡, accum_iter=4：
  参数更新次数/epoch = 600 / 4 = 150 次/epoch

4卡, accum_iter=4：
  参数更新次数/epoch = 150 / 4 = 37.5 次/epoch
```

#### 40 epoch 总步数

| 配置 | steps/epoch | 总步数 (40 epochs) | 参数更新次数 |
|------|-------------|-------------------|--------------|
| 单卡 bz=8 | 600 | 24,000 | 24,000 |
| 4卡 bz=8 | 150 | 6,000 | 6,000 |
| 8卡 bz=8 | 75 | 3,000 | 3,000 |

#### 问题
- [TODO] 当前 accum_iter=1，每步都更新，是否需要调整？
- [TODO] 如果想让参数更新次数更少，可以增大 accum_iter

---

## 模型方面

### 1. 移除 StateGate

#### 当前实现
```python
# shot_adaptation.py
class StateGate(nn.Module):
    def forward(self, q_t):
        alpha = torch.sigmoid(self.gate_mlp(q_t))  # [B, 1, 1]
        return alpha
```

#### 状态更新机制
```
S_t = α * S_{t-1} + (1 - α) * S0
```
- α ≈ 1：保留大部分旧状态（相机连续运动）
- α ≈ 0：重置为初始状态（镜头跳变后）

#### TODO
- [TODO] 暂时移除 StateGate 模块
- [TODO] 简化状态更新逻辑：直接使用 S0（重置）而不经过门控
- [TODO] 保留 ShotTokenGenerator（用于生成 q_t，但不用于门控）
- [TODO] 或者 ShotTokenGenerator 也暂时不用？

#### 影响
- 减少约 98K 参数（StateGate）
- 简化模型逻辑

---

### 2. 将 Residual Adapter 改为 LoRA

#### 当前实现（Residual Adapter）
```python
# PoseResidualAdapter 示例
pose_final = pose_base + gamma * delta_pose
```

#### 目标实现（LoRA）
```python
# 目标：使用 LoRA 形式
# 标准 LoRA: W' = W + BA，其中 B @ A 是低秩分解
# 当前实现更像 residual correction head，而非真正的 LoRA

class PoseLoRAHead(nn.Module):
    def __init__(self, dec_dim=768, rank=8):
        self.gamma = nn.Parameter(torch.tensor(0.0))
        # A: dec_dim × rank, B: rank × output_dim
        self.lora_A = nn.Linear(dec_dim * 2, rank, bias=False)
        self.lora_B = nn.Linear(rank, 7, bias=False)

    def forward(self, z_token, q_out, pose_base):
        # LoRA: delta = B @ A(input)
        x = torch.cat([z_token, q_out], dim=-1)
        delta = self.lora_B(self.lora_A(x))  # [B, 1, 7]

        t_final = pose_base[:, :3] + self.gamma * delta[:, :, :3]
        q_final = F.normalize(pose_base[:, 3:7] + self.gamma * delta[:, :, 3:], dim=-1)
        return torch.cat([t_final, q_final], dim=-1)
```

#### TODO
- [TODO] 确认 LoRA 的 rank 值（建议 r=8 或 r=16）
- [TODO] 对 PoseResidualAdapter → PoseLORALayer
- [TODO] 对 HumanResidualAdapter → HumanLORALayer
- [TODO] 对 WorldResidualAdapter → WorldLORALayer
- [TODO] 保持 gamma 初始化为 0（确保初始状态 = base）

#### LoRA vs Residual Adapter 对比

| 特征 | Residual Adapter | LoRA |
|------|-----------------|------|
| 机制 | 独立 MLP 预测 Δ | 低秩分解 W' = W + BA |
| 参数量 | 较大 (128 hidden) | 较小 (rank × dim) |
| 表达能力 | 较强 | 中等 |
| 正则化效果 | 无 | 较好（低秩约束） |

#### 为什么 LoRA 可能更有用？
1. **更好的正则化**：低秩约束防止过拟合
2. **更少参数**：适合小数据集微调
3. **更稳定**：初始状态更接近原始模型

---

## 总结

### 已确认事项（不改）
- ✅ 分布式框架：DDP + Accelerate，当前规模足够
- ✅ bf16 混合精度：保持 NativeScaler
- ✅ 梯度累积 + 参数更新流程：Accelerate + NativeScaler + AdamW 协同工作，无需改动

### 待决策/待实现
- [TODO] StateGate 移除方案
- [TODO] Residual Adapter → LoRA 的 rank 选择和实现
- [TODO] LoRA 层的具体实现细节

---

## 参考：最终目标模型架构

```
freeze='shot_adaptation' 模式下的可训练模块：

Shot-Aware Adaptation (移除 StateGate 后)
├── ShotTokenGenerator    (~787K) — [TODO] 决定是否保留
├── PoseLORALayer        (~198K) — [TODO] 改 LoRA
├── HumanLORALayer       (~20K)  — [TODO] 改 LoRA
└── WorldLORALayer       (~197K) — [TODO] 改 LoRA

总计: ~1.2M 可训练参数
```
