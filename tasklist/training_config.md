# Movie3R 训练配置

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H800 (80GB) |
| 容器共享内存 | /dev/shm 64MB（不足以支持多进程 DataLoader） |

## 2. 分布式训练框架

### 2.1 当前配置

| 配置 | 实际使用 |
|------|----------|
| 多卡训练 | **DDP** (Accelerate 封装) |
| 混合精度 | **bf16** (Accelerate 内置) |
| 梯度累积 | Accelerate 内置 |
| FSDP | ❌ 没有使用 |
| DeepSpeed | ❌ 没有使用 |

### 2.2 框架关系

```
torchrun 启动多进程
    │
    ▼
Accelerate (高层 API，封装 DDP)
    │
    ├── gradient_accumulation_steps：控制梯度累积
    ├── mixed_precision="bf16"：混合精度
    └── accelerator.prepare()：自动封装 DDP
    │
    ▼
DistributedDataParallel (DDP)
    │
    ├── NCCL backend：多卡通信
    └── 每卡独立模型副本
```

### 2.3 为什么不用 FSDP / DeepSpeed？

| 框架 | 开发方 | 适用场景 |
|------|--------|----------|
| DDP + Accelerate | PyTorch / HuggingFace | 中小模型，配置简单 |
| FSDP | Meta (Facebook) | 超大模型（百亿参数），需要分片模型参数 |
| DeepSpeed | 微软 | 超大模型，ZeRO 优化器（分片 optimizer/gradient/param） |

**Movie3R 场景**：
- 可训练参数 ~1.3M（shot adaptation 模块）
- H800 80GB 显存充足
- DDP + Accelerate **完全够用**，无需更复杂的优化

### 2.4 代码实现

```python
# train.py
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=args.accum_iter,
    mixed_precision="bf16",
    kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
    ],
)

# Accelerate 自动封装 DDP
optimizer, model, data_loader = accelerator.prepare(
    optimizer, model, data_loader
)
```

## 3. 多卡训练

### 3.1 启动方式

使用 `torchrun`（PyTorch 原生分布式）：

```bash
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE}
```

### 3.2 关键点
- `batch_size` 是**每卡**的 batch size，不是全局 batch size
- 全局 batch size = `batch_size × num_gpus`
- torchrun + Accelerate 自动处理 NCCL 通信

## 4. Batch Size 与 num_views

### 4.1 关键配置：num_views=4

```yaml
# train.yaml
num_views: 4                    # 每个 sample 包含 4 个视角
img_size: (512, 512)            # 主图像尺寸
mhmr_img_res: 896               # Multi-HMR 部分处理 896×896
```

**重要**：`batch_size=1` 实际意味着 **1 个 sample × 4 个视角 = 4 张图同时在显存里**

```
batch_size=1 的实际处理：
├── sample 1
│   ├── view 1: 512×512 图像
│   ├── view 2: 512×512 图像
│   ├── view 3: 512×512 图像
│   └── view 4: 512×512 图像
│
└── 这 4 张图会同时经过 encoder/decoder，产生大量激活值
```

### 4.2 显存占用分析

| 组件 | 显存占用 | 说明 |
|------|----------|------|
| 模型权重 (bf16) | ~5GB | encoder + decoder + backbone + heads |
| 4视角图像输入 | ~4MB | 可忽略 |
| Encoder 激活值 | ~20GB | ViT 输出，4 张图同时在显存 |
| Decoder 激活值 | ~15GB | cross-attention，视角间交互 |
| 梯度 + 优化器状态 | ~10GB | 训练模式 |
| **总计** | **~50GB** | |

### 4.3 单卡 Batch Size 测试结果（H800 80GB, freeze='shot_adaptation'）

| batch_size | 实际图片数 | 显存使用 | 状态 |
|------------|-----------|----------|------|
| 1 | 4 | ~46GB | ✅ OK |
| 2 | 8 | ~48GB | ✅ OK |
| 4 | 16 | ~48GB | ✅ OK |
| 8 | 32 | ~53GB | ✅ OK |

**单卡最大 batch_size = 8**（gradient_checkpointing=true 时）

> 注：batch_size 从 1→2→4 显存增长很小，是因为 gradient_checkpointing 生效，用计算换显存

### 4.4 Batch Size 选择建议

| batch_size | 场景 |
|------------|------|
| 8 | 单卡推荐，充分利用显存 |
| 4 | 显存不足时的备用选项 |
| 2 | 保守配置，显存紧张时使用 |

### 4.5 多卡配置

| 配置 | 命令 | 每卡 batch | 总 batch |
|------|------|-----------|----------|
| 单卡 | `./train.sh 1 40 8` | 8 | 8 |
| 4卡 | `./train.sh 4 40 8` | 8 | 32 |
| 8卡 | `./train.sh 8 40 8` | 8 | 64 |

### 4.6 梯度累积

当需要更大等效 batch size 但显存受限时：

```yaml
batch_size: 8
accum_iter: 4    # 等效 batch = 8 × 4 × num_gpus
```

梯度累积由 **Accelerate** 库的 `accelerator.accumulate(model)` 自动处理，不是手写。

**原理**：
```python
for batch in dataloader:
    loss.backward()  # 梯度累加到 .grad（每个 batch 都执行）

    if step % accum_iter == 0:
        optimizer.step()     # 用累积的梯度更新
        optimizer.zero_grad() # 清空
```

batch_size=8 时通常不需要梯度累积。

## 5. train.sh 脚本参数

```bash
./train.sh [num_gpus] [epochs] [batch_size]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| num_gpus | GPU 数量，"auto" 自动检测空闲卡 | auto |
| epochs | 训练轮数 | 1 |
| batch_size | 每卡 batch size | 1 |

**使用示例**：
```bash
./train.sh        # 自动检测空闲卡
./train.sh 1 40 8 # 单卡训练
./train.sh 4 40 8 # 4卡训练
./train.sh 0      # 只查看 GPU 状态
```

## 6. train.yaml 关键配置

```yaml
# 训练
batch_size: 8
accum_iter: 1
gradient_checkpointing: true
amp: 1                     # bf16 混合精度

# 数据加载
num_workers: 0             # 重要：避免 /dev/shm 不足

# 分布式
dist_backend: 'nccl'
```

## 7. NativeScaler 与 bf16

### 7.1 作用

```python
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
loss_scaler = NativeScaler(accelerator=accelerator)
```

解决 **bf16 梯度下溢** 问题：
- bf16 表示范围有限，loss 很小时梯度可能变 0
- Loss Scaler 放大 loss，计算后再还原

### 7.2 与梯度累积的关系

| 功能 | 实现 |
|------|------|
| 梯度累积 | `accelerator.accumulate(model)` |
| bf16 Loss Scaling | `NativeScaler` |

两者独立，协同工作。

## 8. 常见问题

### 8.1 NCCL 多卡通信问题
- 检查 NCCL 版本
- 尝试 `GLOO_SOCKET_IFNAME=lo` 环境变量

### 8.2 /dev/shm 不足
解决：使用 `num_workers=0` 单进程模式
