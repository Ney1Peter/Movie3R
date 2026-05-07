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
- ✅ `forward_recurrent_lighter` 已接入 ShotToken/LoRA 推理路径
- 🚨 LoRA64 正式训练权重推理失败：打开 `enable_shot_adaptation` 后相机/人体/pointmap 尺度崩坏

#### LoRA 架构

| 模块 | rank=64 参数量 | rank=128 参数量 |
|------|---------------|-----------------|
| ShotTokenGenerator | 788K | 788K |
| PoseLoRALayer | 99K | 198K |
| HumanLoRALayer | 98K | 197K |
| WorldLoRALayer | 98K | 197K |
| **总计** | **1.08M** | **1.38M** |

#### LoRA Head V1 修正范围（当前位置/朝向优先）

| 模块 | 修正内容 | 不修正内容 |
|------|----------|------------|
| PoseLoRALayer | `camera_pose` 的 translation + quaternion | - |
| HumanLoRALayer | `smpl_transl` | `smpl_shape` / `smpl_rotmat` / `smpl_expression` |
| WorldLoRALayer | `pts3d_in_self_view` 和 `pts3d_in_other_view` 的全局 3D shift | 局部几何细节 |

**说明**：V1 目标是修正镜头跳变带来的位置/朝向偏移，不修改人体形状、姿态细节、表情，也不做局部 pointmap 自由形变。

#### LoRA Head V1 当前可训练参数估算（rank=128）

| 模块 | 当前作用 | 参数量估算 |
|------|----------|------------|
| ShotTokenGenerator | 生成 shot token | ~788K |
| PoseLoRALayer | 修正 `camera_pose` | ~198K |
| HumanLoRALayer | 只修正 `smpl_transl` | ~197K |
| WorldLoRALayer | 修正 pointmap 全局 shift | ~197K |
| **总计** | | **~1.38M** |

#### shot_label 使用策略

- ✅ V1 暂不使用 `shot_label` 作为显式监督
- ✅ 当前通过相邻帧 image token 差异生成 `q_t`，再由最终 task loss 隐式学习何时需要修正
- ⚠️ 当前 `q_t` 更准确是 shot-conditioned adaptation token，不是显式 shot-change classifier
- [TODO] V2 可增加 `shot_logit` + `shot_label` BCE 辅助 loss
- [TODO] 如后续重新加入 StateGate，可用 shot probability 控制 state mixing

---

### 3. Shot Adaptation 推理消融问题 🚨

#### 问题描述
- **Training 路径** (`_forward_impl`)：✅ 使用 ShotToken/LoRA
- **Inference 路径** (`forward_recurrent_lighter`)：✅ 已接入 ShotToken/LoRA
- **当前问题**：LoRA64 `checkpoint-best.pth` 推理视觉结果严重错误，相机尺度异常、人很小、场景乱成一团

#### 影响
- 训练 loss 和 AvatarReX val/test loss 下降，但 demo 视觉质量崩坏
- 说明当前指标不能充分约束 demo 关心的绝对尺度与可视化质量
- 需要先验证 shot token 本身和数据集 `shot_label` 质量，再决定是否继续训练或调整架构

#### 已完成消融

同一段 `data/h36.mp4` 前 8 帧：

| 模式 | camera 平移均值 | pointmap 范围均值 | SMPL 平移均值 | 结论 |
|------|----------------|------------------|---------------|------|
| base Human3R | `0.010` | `9.049` | `4.935` | 原模型正常 |
| LoRA64 checkpoint，关闭 `enable_shot_adaptation` | `0.010` | `9.049` | `4.935` | base 权重正常，checkpoint 加载正常 |
| LoRA64 checkpoint，打开 `enable_shot_adaptation` | `0.042` | `3.844` | `3.067` | shot adaptation 分支破坏尺度 |
| LoRA64 checkpoint，LoRA gamma 全置 0，仅保留 trained shot token | `0.020` | `3.844` | `1.966` | pointmap 崩坏主要来自 trained shot token 进入 decoder |

#### 当前判断
- `enable_shot_adaptation=False` 不是恢复 checkpoint，而是推理时跳过新增 shot/LoRA 分支
- 因为训练时 base Human3R 参数被冻结，所以关闭分支后基本等价于原 Human3R
- `q_t` 直接 append 到 frozen decoder token 序列不是 residual-safe；即使 LoRA gamma=0，额外 token 仍会通过 decoder attention 改变所有输出 token

#### 待解决
- [TODO] 验证 `AvatarReX_Video` / `AvatarReX_AABB` 的 `shot_label` 是否符合预期
- [TODO] 统计训练前 `g_curr/g_prev` 的 cosine similarity 和差异范数，看是否天然区分连续/跳变
- [TODO] 统计训练前/训练后 `q_t` 的范数、cosine、连续/跳变可分性
- [TODO] 增加可视化/诊断指标：camera translation norm、pointmap extent、SMPL translation norm、human/scene scale ratio
- [TODO] 修复 val/test dataset key 重复/覆盖问题
- [TODO] 重新评估 `q_t` 进入 decoder 的注入方式，避免无约束破坏 base 输出

---

### 4. Shot Token 质量验证计划

#### 当前验证结果（2026/05/07）

已新增脚本：`scripts/analyze_shot_token.py`

| 层级 | 结果 | 结论 |
|------|------|------|
| 数据集 `shot_label` | 3 个 root 每类抽样 20 个，Video 全 `[0,0,0,0]`，AABB 全 `[0,0,1,0]`，invalid=0 | ✅ 数据格式正确 |
| 输入特征 `g_curr/g_prev` | `g_diff_norm` 连续约 `0.697`，跳变约 `2.445`，AUC `0.9997` | ✅ 输入特征足够区分跳变 |
| 训练后 `q_t` 范数 | 连续约 `62.17`，跳变约 `62.21`，AUC `0.51` | ❌ 范数不能区分跳变且幅度失控 |
| view2 `q_delta_norm` | 连续约 `1.21`，跳变约 `5.22` | ⚠️ 有跳变响应，但不是 no-op 安全 |
| 相对 decoder token 尺度 | `q_t` norm 约为 decoder image token norm 的 `2.75x` | ❌ prompt 过强，容易扰动 frozen decoder |

当前判断：数据集和输入特征没有明显问题；ShotTokenGenerator 不是完全无效，但训练出的 `q_t` 过强、缺少连续帧 no-op/gating/范数约束，是这次推理崩坏的主要风险点。

补充解释：`cosine(g_curr, g_prev)` 的 `0.9990` 和 `0.9889` 虽然都接近 1，但高维特征中该差异很稳定；结合 `diff_norm` 跳变帧约为连续帧 3.5 倍，以及 AUC `0.9997`，可以判断输入特征本身可分。no-op 指连续帧时 shot token 应基本“不操作”，不明显改变原 Human3R 输出。

#### 下一版约束方案

推荐结构：

```text
q_raw, shot_logit = ShotTokenGenerator(g_curr, g_prev, diff, sim)
shot_prob = sigmoid(shot_logit)
q_t = shot_scale * shot_prob * LayerNorm(q_raw)
```

优先实现：

| 约束 | 目的 |
|------|------|
| `shot_logit + BCE(shot_label)` | 显式监督 shot/change 判断 |
| `shot_prob` gate | 连续帧弱化 `q_t`，跳变帧增强 `q_t` |
| `LayerNorm(q_raw)` | 限制 `q_t` 尺度 |
| `shot_scale` 初始接近 0 | 保持训练初期接近原 Human3R |
| `(1-shot_label) * ||q_t||^2` | 连续帧 no-op 正则 |
| 输出尺度监控 | 防止 loss 下降但推理尺度崩坏 |

可选加强：连续帧输出级 no-op loss。

```text
L_noop = (1 - shot_label) * ||pred_with_shot - stopgrad(pred_without_shot)||
```

#### 验证目标

在继续训练前，先判断问题来自哪里：

| 层级 | 要回答的问题 | 通过标准 |
|------|--------------|----------|
| 数据集 | `shot_label` 是否正确 | Video 全 0，AABB 为 `[0, 0, 1, 0]` |
| 特征 | `g_curr/g_prev` 是否能区分跳变 | AABB 跳变帧 cosine 更低、diff norm 更高 |
| ShotTokenGenerator | `q_t` 是否有区分度 | 连续/跳变 `q_t` 分布可分，AUC 明显高于随机 |
| 注入方式 | `q_t` 是否破坏 base | 未训练/小幅 `q_t` 注入不应大幅改变 pointmap/camera/SMPL 尺度 |

#### 建议诊断统计

- `shot_label`
- `cosine_similarity(g_curr, g_prev)`
- `||g_curr - g_prev||`
- `||q_t||`
- `cosine_similarity(q_t, q_{t-1})`
- 打开/关闭 `enable_shot_adaptation` 的输出差异
- pointmap extent、camera translation norm、SMPL translation norm

#### 决策规则

- 如果 `shot_label` 错：先修数据集
- 如果 `g_curr/g_prev` 不可分：改 shot token 输入特征
- 如果 `g` 可分但 `q_t` 不可分：给 ShotTokenGenerator 加 `shot_logit` + BCE 辅助监督
- 如果 `q_t` 可分但输出崩：改 `q_t` 注入 decoder 的方式或增加 no-op/scale 约束

---

## 总结

### 已确认事项
- ✅ 分布式框架：DDP + Accelerate，当前规模足够
- ✅ bf16 混合精度：保持 NativeScaler
- ✅ 梯度累积 + 参数更新流程：无需改动
- ✅ LoRA 训练路径工作正常
- ✅ Inference 已接入 ShotToken/LoRA
- ✅ 数据集 `shot_label` 格式检查通过
- ✅ `g_curr/g_prev` 输入特征可高质量区分连续/跳变
- ❌ LoRA64 `checkpoint-best.pth` 推理失败，问题集中在 shot adaptation 分支
- ❌ 训练后 `q_t` 幅度过强，连续帧也不是 no-op
- ❌ 当前 loss/val 指标不能充分代表 demo 视觉质量

### 待决策/待实现
- [TODO] Layer 1：给 `ShotTokenGenerator` 增加 `shot_logit` + `shot_label` BCE 辅助监督
- [TODO] Layer 2：给 `q_t` 增加 `shot_prob` gate / `LayerNorm` / `shot_scale`
- [TODO] Layer 3：增加连续帧 no-op 输出约束，保护 base Human3R 行为
- [TODO] 增加 `q_norm`、`q_delta_norm`、`shot_auc` 和输出尺度监控指标

### 分层提交要求
- Layer 1/2/3 每层先做“原代码注释备份”并单独 commit
- 每层备份 commit 后，再新增实现并单独 commit
- 不把备份和新实现压到同一个 commit
- 不主动 push
