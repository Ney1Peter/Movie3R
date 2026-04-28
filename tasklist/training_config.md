# Human3R 训练配置详解

## 1. 硬件与训练环境

### 1.1 硬件规格
| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H800 (80GB) |
| 显存 | 80GB per card |
| 容器共享内存 | /dev/shm 64MB (不足以支持多进程 DataLoader) |

### 1.2 关键环境变量
```bash
export TORCH_HOME=$HOME/.cache/torch
export TORCH_HUB_USE_HEURISTICS=0
```

---

## 2. 多卡训练机制

### 2.1 多卡实现方式
使用 `torchrun` (PyTorch 原生分布式训练)：

```bash
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE} \
    ...
```

### 2.2 数据并行原理

```
                          ┌─────────────────────────────────────┐
                          │           torchrun                   │
                          │  (自动启动 N 个进程，每进程一个 GPU)  │
                          └─────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
        ┌───────────┐              ┌───────────┐              ┌───────────┐
        │  GPU 0    │              │  GPU 1    │      ...     │  GPU N-1  │
        │ Process 0 │              │ Process 1 │              │ Process N-1│
        └───────────┘              └───────────┘              └───────────┘
              │                           │                           │
              ▼                           ▼                           ▼
        DataLoader                  DataLoader                  DataLoader
        (每个 GPU 独立加载)          (每个 GPU 独立加载)
              │
              ▼
        每卡 batch_size 个样本
```

**关键点**:
- `batch_size` 是**每卡**的 batch size，不是全局 batch size
- 全局 batch size = `batch_size × num_gpus`
- torchrun 自动处理进程间通信 (NCCL)

### 2.3 单卡 vs 多卡配置

| 配置 | 命令 | batch_size | 等效全局 batch |
|------|------|------------|----------------|
| 单卡 | `python train.py` | 2 | 2 |
| 4卡 | `torchrun --nproc_per_node=4 ...` | 2 | 8 |
| 8卡 | `torchrun --nproc_per_node=8 ...` | 2 | 16 |

---

## 3. Batch Size 与显存

### 3.1 单卡 Batch Size 测试结果

**测试环境**: H800 80GB, freeze='shot_adaptation' 模式

| batch_size | 显存使用 | 每 epoch 时间 | 状态 |
|------------|----------|---------------|------|
| 1 | ~46GB | ~4h | ✅ OK |
| 2 | ~48GB | ~2.5h | ✅ OK |
| 4 | ~48GB | ~2.5h | ✅ OK |
| 8 | ~53GB | ~2h | ✅ OK |

**结论**: 单卡最大 batch_size = 8（受限于 H800 80GB 显存）

### 3.2 Batch Size 与显存关系

```
batch_size=1: ~46GB  (基础显存占用)
batch_size=2: ~48GB  (+2GB)
batch_size=4: ~48GB  (几乎不增加，gradient checkpointing 生效)
batch_size=8: ~53GB   (+5GB)
```

**注意**: 使用 `gradient_checkpointing=true` 时，batch_size 从 2 到 4 显存几乎不增加，因为 checkpointing 用计算换显存。

### 3.3 Batch Size 选择建议

**场景 1: 单卡训练（推荐）**
```bash
./train.sh 1 40 8    # 1卡, 40 epochs, batch_size=8
```

**场景 2: 4卡训练**
```bash
./train.sh 4 40 8    # 4卡, 40 epochs, batch_size=8 per GPU
                        # 等效全局 batch_size = 32
```

**场景 3: 8卡训练**
```bash
./train.sh 8 40 8    # 8卡, 40 epochs, batch_size=8 per GPU
                        # 等效全局 batch_size = 64
```

**推荐配置**: 单卡 batch_size=8，训练时间约 2h/epoch

### 3.4 梯度累积 (accum_iter)

当需要更大的等效 batch size 但显存受限时，可使用梯度累积：

```yaml
batch_size: 8
accum_iter: 4    # 累积4个 batch 再更新梯度
# 等效 batch_size = 8 × 4 = 32
```

**注意**: 使用 batch_size=8 时通常不需要梯度累积，直接用更大 batch 效率更高。

**原理**:
```
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accum_iter    # 缩放 loss
    loss.backward()             # 累积梯度

    if (i + 1) % accum_iter == 0:
        optimizer.step()        # 更新参数
        optimizer.zero_grad()  # 清空梯度
```

**训练时间影响**: 梯度累积不会显著增加训练时间，因为仍需执行前向传播。

---

## 4. 数据集配置

### 4.1 AvatarReX 数据集划分

| Split | 配置 | 样本数 | Seed |
|-------|------|--------|------|
| train | 800 × 6 datasets | 4800 | 11 |
| val | 100 × 6 datasets | 600 | 22 |
| test | 100 × 6 datasets | 600 | 33 |

**训练集组成**:
```
train = 800 @ AvatarReX_Video(zzr)   # Video, seed=11
      + 800 @ AvatarReX_Video(lbn1)  # Video, seed=11
      + 800 @ AvatarReX_Video(zxc)   # Video, seed=11
      + 800 @ AvatarReX_AABB(zzr)    # AABB, seed=11
      + 800 @ AvatarReX_AABB(lbn1)   # AABB, seed=11
      + 800 @ AvatarReX_AABB(zxc)    # AABB, seed=11
= 4800 samples/epoch
```

### 4.2 分辨率配置

训练时使用多分辨率数据增强：
```python
resolution = [
    (512, 384), (512, 336), (512, 288), (512, 256), (512, 208), (512, 144),
    (384, 512), (336, 512), (288, 512), (256, 512),
]
```

---

## 5. 模型配置

### 5.1 Model Architecture

```yaml
model: ARCroco3DStereo(
    freeze='shot_adaptation',    # 新配置：只训练 ~1.3M 新参数
    state_size=768,
    state_pe='2d',
    pos_embed='RoPE100',
    rgb_head=True,
    pose_head=True,
    msk_head=True,
    img_size=(512, 512),
    head_type='dpt',
    output_mode='pts3d+pose+smpl',
    enc_embed_dim=1024,
    enc_depth=24,
    dec_embed_dim=768,
    dec_depth=12,
    backbone='dinov2_vitl14',
)
```

### 5.2 freeze='shot_adaptation' 训练参数

| 模块 | 参数量 | 训练状态 |
|------|--------|----------|
| ShotTokenGenerator | ~787K | ✅ 训练 |
| StateGate | ~99K | ✅ 训练 |
| LoRAPoseHead | ~198K | ✅ 训练 |
| LoRAHumanHead | ~20K | ✅ 训练 |
| LoRAWorldGlobalShift | ~197K | ✅ 训练 |
| **新增模块总计** | **~1.3M** | ✅ 训练 |
| encoder (ViT) | ~600M | ❌ 冻结 |
| decoder | ~226M | ❌ 冻结 |
| backbone (Dinov2) | 304M | ❌ 冻结 |
| downstream_head | 152M | ❌ 冻结 |

**优势**: 只训练 0.1% 的参数，显存占用大幅降低

### 5.3 显存预估 (freeze='shot_adaptation')

由于大部分参数冻结，显存占用应该比 freeze='none' 小很多：

| 配置 | 预估显存 |
|------|----------|
| freeze='none', batch_size=1 | ~51GB |
| freeze='none', batch_size=2 | ~53GB |
| freeze='shot_adaptation', batch_size=2 | < 40GB (预估) |
| freeze='shot_adaptation', batch_size=4 | 可能可行 (待测试) |

---

## 6. 优化器配置

### 6.1 学习率调度

```yaml
lr: 1.0e-04        # 基础学习率
min_lr: 1.0e-06    # 最小学习率
warmup_epochs: 5   # Warmup epochs
weight_decay: 0.05
```

**学习率调度策略**:
- 前 5 epochs 进行 warmup
- 之后使用余弦衰减到 min_lr

### 6.2 优化器参数组

当前使用统一的 lr=1e-4，没有对不同模块设置不同学习率：

```python
param_groups = misc.get_parameter_groups(model, args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
```

**注意**: shot_adaptation 模块和其他冻结模块使用相同的学习率。如果需要差异化学习率，需要修改代码。

---

## 7. 完整训练参数

### 7.1 train.yaml 关键配置

```yaml
# 模型
model: ARCroco3DStereo(ARCroco3DStereoConfig(freeze='shot_adaptation', ...))
pretrained: /workspace/code/Movie3R/src/human3r_896L.pth

# 训练
batch_size: 8              # ⚠️ 需要根据卡数调整
accum_iter: 1              # 梯度累积
gradient_checkpointing: true
epochs: 40
weight_decay: 0.05
lr: 1.0e-04
min_lr: 1.0e-06
warmup_epochs: 5
amp: 1                     # 混合精度训练

# 数据加载
num_workers: 0             # 重要：避免 /dev/shm 不足

# 分布式
dist_url: 'env://'
dist_backend: 'nccl'

# 验证与保存
early_stopping_patience: 10
eval_freq: 1
save_freq: 1
print_freq: 100
```

---

## 8. 训练脚本参数

### 8.1 train.sh 参数说明

```bash
./train.sh [num_gpus] [epochs] [batch_size]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| num_gpus | GPU 数量，或 "auto" 自动检测空闲卡 | auto |
| epochs | 训练轮数 | 1 |
| batch_size | 每卡 batch size | 1 |

### 8.2 自动检测空闲 GPU

脚本会自动检测显存使用 < 2GB 的 GPU 为空闲卡：

```bash
./train.sh        # 自动检测所有空闲卡
./train.sh 4      # 使用 4 张卡
./train.sh 0      # 只查看 GPU 状态
```

---

## 9. 训练输出

### 9.1 输出目录结构

```
experiments/
  avatarrex_zzr_lbn1/
    checkpoint-final.pth      # 最终模型 (仅权重, ~4.7GB)
    checkpoint-last.pth       # 最后一个 epoch (含优化器, ~11.5GB)
    checkpoint-best.pth       # 最佳验证模型 (含优化器, ~11.5GB)
    logs/
      events.*               # TensorBoard 日志
```

### 9.2 监控指标

| 指标 | 说明 |
|------|------|
| Total Loss | 总损失 |
| Regr3DPoseBatchList_self_pts3d | 深度点损失 |
| pose_loss | 位姿损失 |
| pose_loss_view2_AABB | AABB 跳变帧位姿损失 |
| SMPLLoss_* | SMPL 参数损失 |
| lr | 当前学习率 |

---

## 10. 推荐训练配置

### 10.1 Shot Adaptation 训练 (推荐)

```bash
# 4卡训练，batch_size=2 per GPU
./train.sh 4 40 2
```

**配置**:
- GPU: 4 × H800
- batch_size per GPU: 2
- 等效全局 batch: 8
- epochs: 40
- 预估训练时间: ~20-30 小时

### 10.2 如果显存允许更大 batch

```bash
# 4卡训练，batch_size=4 per GPU (如果 freeze='shot_adaptation' 显存够用)
./train.sh 4 40 4
```

**配置**:
- GPU: 4 × H800
- batch_size per GPU: 4
- 等效全局 batch: 16
- epochs: 40

### 10.3 梯度累积配置 (如果需要更大等效 batch)

```yaml
# train.yaml
batch_size: 2
accum_iter: 4    # 等效全局 batch = 2 × 4 × num_gpus
```

---

## 11. 常见问题

### 11.1 NCCL 多卡通信问题
如果多卡训练卡住，参考 Section 16 排查：
- 检查 NCCL 版本
- 尝试使用 Gloo backend (速度较慢)

### 11.2 /dev/shm 不足
解决：使用 `num_workers=0` 单进程模式

### 11.3 Batch Size 选择
- 单卡最大: batch_size=2
- 多卡时: batch_size 是每卡的数量，不是全局数量
- 等效全局 batch = batch_size × num_gpus × accum_iter
