# Movie3R 训练配置

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H800 (80GB) |
| 容器共享内存 | /dev/shm 64MB（不足以支持多进程 DataLoader） |

## 2. 多卡训练

### 2.1 启动方式
使用 `torchrun`（PyTorch 原生分布式）：

```bash
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE}
```

### 2.2 关键点
- `batch_size` 是**每卡**的 batch size，不是全局 batch size
- 全局 batch size = `batch_size × num_gpus`
- torchrun 自动处理进程间通信 (NCCL)

## 3. Batch Size 配置

### 3.1 单卡 Batch Size 测试结果（H800 80GB, freeze='shot_adaptation'）

| batch_size | 显存使用 | 状态 |
|------------|----------|------|
| 1 | ~46GB | ✅ OK |
| 2 | ~48GB | ✅ OK |
| 4 | ~48GB | ✅ OK |
| 8 | ~53GB | ✅ OK |

**单卡最大 batch_size = 8**（gradient_checkpointing=true 时）

### 3.2 Batch Size 选择建议

| batch_size | 场景 |
|------------|------|
| 8 | 单卡推荐，充分利用显存 |
| 4 | 显存不足时的备用选项 |
| 2 | 保守配置，显存紧张时使用 |

### 3.3 多卡配置

| 配置 | 命令 | 每卡 batch | 总 batch |
|------|------|-----------|----------|
| 单卡 | `./train.sh 1 40 8` | 8 | 8 |
| 4卡 | `./train.sh 4 40 8` | 8 | 32 |
| 8卡 | `./train.sh 8 40 8` | 8 | 64 |

### 3.4 梯度累积

当需要更大等效 batch size 但显存受限时：

```yaml
batch_size: 8
accum_iter: 4    # 等效 batch = 8 × 4 × num_gpus
```

batch_size=8 时通常不需要梯度累积。

## 4. train.sh 脚本参数

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

## 5. train.yaml 关键配置

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

## 6. 常见问题

### 6.1 NCCL 多卡通信问题
- 检查 NCCL 版本
- 尝试 `GLOO_SOCKET_IFNAME=lo` 环境变量

### 6.2 /dev/shm 不足
解决：使用 `num_workers=0` 单进程模式
