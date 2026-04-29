# Movie3R

> 基于 Human3R 的多镜头电影级人体重建扩展，解决镜头跳变（shot change）带来的时序不连续问题。

[![arXiv](https://img.shields.io/badge/Arxiv-2510.06219-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.06219)
[![Project Page](https://img.shields.io/badge/Project-Human3R-C27185.svg)](https://fanegg.github.io/Human3R)

---

## 背景与动机

### 问题

电影级多镜头人体重建面临独特挑战：**镜头跳变（shot change）**导致时序不连续：
- 相机视角突然切换（如切到不同机位）
- 旧帧的时序信息在新镜头下不再有效
- 直接使用会导致错误累积

### 解决方案

Movie3R 在 Human3R 基础上引入 **Shot-Aware Adaptation** 机制：

1. **ShotTokenGenerator**：检测相邻帧的不连续程度
2. **StateGate**：软性门控状态更新（镜头跳变时快速重置）
3. **Residual Adapter**：对 base model 输出做轻量微调修正

### 优势

| 特性 | 说明 |
|------|------|
| 只训练 ~1.3M 参数 | 占全部参数的 0.1% |
| 不破坏预训练 | gamma 初始化为 0，初始状态 = base model |
| 快速适应镜头跳变 | 轻量修正，推理零额外开销 |

---

## 项目结构

```
Movie3R/
├── src/
│   ├── train.py              # 训练入口
│   ├── demo.py               # 推理演示
│   ├── dust3r/
│   │   ├── model.py          # 模型定义 (ARCroco3DStereo)
│   │   └── shot_adaptation.py # Shot-Aware Adaptation 模块
│   └── croco/                # 底层 CroCo 模型
├── config/
│   └── train.yaml            # 训练配置
├── scripts/
│   └── train.sh              # 训练启动脚本
├── docs/
│   ├── movie3r/              # Movie3R 文档
│   ├── train_code_explanation.md  # 训练代码详解
│   └── env_setup.md          # 环境配置
└── tasklist/
    └── TODO.md               # 待实现功能
```

---

## 快速开始

### 环境安装

详见 [环境配置文档](docs/env_setup.md)

```bash
# 1. 创建环境
uv venv .venv --python 3.10
source .venv/bin/activate

# 2. 安装依赖
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_Movie3R.txt

# 3. 编译 RoPE
cd src/croco/models/curope
python setup.py build_ext --inplace
```

### 启动训练

```bash
cd src

# 单卡训练
./train.sh 1 40 8

# 4卡训练
./train.sh 4 40 8

# 参数说明
./train.sh [num_gpus] [epochs] [batch_size]
```

### 模型推理

```bash
python demo.py --model_path src/human3r_896L.pth \
    --seq_path examples/video.mp4 --output_dir output
```

---

## 核心模块

### Shot-Aware Adaptation

详见：[模型架构文档](docs/movie3r/model.md)

```
输入帧 i-1                     输入帧 i
    │                             │
    ▼                             ▼
encoder + decoder             encoder + decoder
    │                             │
    │◄─────────────────────────────┤
    │        ShotTokenGenerator    │
    │           q_t = f(feat_i, feat_{i-1})
    │                             │
    ▼                             ▼
StateGate(q_t)               StateGate(q_t)
  α = sigmoid(MLP(q_t))        α = sigmoid(MLP(q_t))
    │                             │
    ▼                             ▼
S_t = α*S_prev + (1-α)*S0    S_t = α*S_prev + (1-α)*S0
    │                             │
    ├─────────────────────────────►┤
    ▼
Residual Adapters (Pose / Human / World)
    │
    ▼
Final Predictions
```

### 可训练参数

| 模块 | 参数量 | 说明 |
|------|--------|------|
| ShotTokenGenerator | ~787K | 生成 shot token |
| StateGate | ~99K | 状态门控 |
| PoseResidualAdapter | ~198K | 位姿修正 |
| HumanResidualAdapter | ~20K | SMPL 参数修正 |
| WorldResidualAdapter | ~197K | 场景点云修正 |
| **总计** | **~1.3M** | 占 0.1% |

---

## 训练配置

详见：[训练配置文档](docs/movie3r/training.md)

### 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H800 (80GB) |
| batch_size | 单卡最大 8 |
| 混合精度 | bf16 |

### 分布式训练

- 框架：`torchrun` + `Accelerate`（封装 DDP）
- 底层通信：NCCL
- 当前不使用 FSDP/DeepSpeed（规模不需要）

### Batch Size 与显存

| batch_size | 显存使用 | 图片数 |
|------------|----------|--------|
| 1 | ~46GB | 4 |
| 2 | ~48GB | 8 |
| 4 | ~48GB | 16 |
| 8 | ~53GB | 32 |

> 注：`num_views=4`，所以 batch_size=1 实际处理 4 张图

---

## 文档索引

### 开发文档

| 文档 | 内容 |
|------|------|
| [训练代码详解](docs/train_code_explanation.md) | train.py 完整流程解析 |
| [训练配置](docs/movie3r/training.md) | 硬件、Batch Size、分布式配置 |
| [模型架构](docs/movie3r/model.md) | Shot-Aware Adaptation 设计 |
| [TODO](tasklist/TODO.md) | 待实现功能清单 |

### 环境配置

| 文档 | 内容 |
|------|------|
| [环境配置](docs/env_setup.md) | Python 环境、依赖、RoPE 编译 |
| [环境变量](docs/env_setup.md#环境变量) | TORCH_HOME, etc. |

### 参考文档

| 文档 | 内容 |
|------|------|
| [Human3R 训练](docs/train.md) | 原版 Human3R 训练流程 |
| [Human3R 评估](docs/eval.md) | 模型评估方法 |
| [Human3R 推理](docs/inference.md) | 不同 backbone 使用 |

---

## 许可证

本项目基于 [Human3R](https://github.com/fanegg/Human3R) 开发，遵循其许可证。

---

## 致谢

我们的代码基于以下开源项目：

- [Human3R](https://github.com/fanegg/Human3R)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [Multi-HMR](https://github.com/naver/multi-hmr)
- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
