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

#### 决策
- ✅ 对于当前规模（1.3M 参数），DDP + Accelerate 足够
- ✅ 如后续扩展到超大模型，再考虑 FSDP/DeepSpeed

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

单卡训练：600 steps/epoch
4卡训练：150 steps/epoch
8卡训练：75 steps/epoch
```

---

## 模型方面

### 1. 移除 StateGate

#### 状态
- ✅ StateGate 已移除
- ~~✅ 状态更新改为直接使用 S0 重置~~（已废弃，不符合当前训练设计）
- ✅ 状态更新已恢复为原 Human3R 行为：默认继续使用前一帧 recurrent state
- 减少约 98K 参数

---

### 2. LoRA 实现

#### 当前状态
- ✅ LoRA rank=64 实现完成
- ✅ LoRA rank=128 实现完成
- ✅ LoRA Head V1 范围收窄：优先修正位置/朝向，不修人体细节和表情
- ⚠️ **Inference 路径不支持 LoRA** (问题)

#### LoRA 架构

| 模块 | rank=64 参数量 | rank=128 参数量 |
|------|---------------|-----------------|
| PoseLoRALayer | 99K | 132K |
| HumanLoRALayer | 197K | 264K |
| WorldLoRALayer | 98K | 131K |
| ShotTokenGenerator | 526K | 526K |
| **总计** | **789K** | **1,053K** |

#### LoRA Head V1 修正范围（当前位置/朝向优先）

| 模块 | 修正内容 | 不修正内容 |
|------|----------|------------|
| PoseLoRALayer | `camera_pose` 的 translation + quaternion | - |
| HumanLoRALayer | `smpl_transl` | `smpl_shape` / `smpl_rotmat` / `smpl_expression` |
| WorldLoRALayer | `pts3d_in_self_view` 和 `pts3d_in_other_view` 的全局 3D shift | 局部几何细节 |

**说明**：V1 目标是修正镜头跳变带来的位置/朝向偏移，不修改人体形状、姿态细节、表情，也不做局部 pointmap 自由形变。

---

### 3. Inference LoRA 支持问题 🚨

#### 问题描述
- **Training 路径** (`_forward_impl`)：✅ 使用 LoRA
- **Inference 路径** (`forward_recurrent_lighter`)：❌ 不使用 LoRA

#### 影响
- 训练时 LoRA 正常更新参数
- 但推理时完全不走 LoRA 路径
- **导致训练效果无法在推理时体现**

#### 修复尝试
1. 在 `forward_recurrent_lighter` 中添加 shot token 生成
2. 修改 token slicing 以支持 q' 在末尾
3. 添加 LoRA application

**结果**：❌ 推理结果完全错误，已回滚

#### 待解决
- [TODO] 分析原始 Human3R inference 架构
- [TODO] 确定正确的 inference + LoRA 方案
- [TODO] 实现修复

---

## 总结

### 已确认事项
- ✅ 分布式框架：DDP + Accelerate，当前规模足够
- ✅ bf16 混合精度：保持 NativeScaler
- ✅ 梯度累积 + 参数更新流程：无需改动
- ✅ LoRA 训练路径工作正常
- ❌ **Inference 不支持 LoRA**（待解决）

### 待决策/待实现
- [TODO] **Inference LoRA 支持**（优先级最高）
- [TODO] 验证 LoRA 在 inference 时生效
