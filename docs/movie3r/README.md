# Movie3R

Movie3R 是基于 Human3R 的扩展，针对**多镜头电影级人体重建**场景优化，核心解决**镜头跳变（shot change）**带来的时序不连续问题。

## 快速开始

### 环境安装

详见 [环境配置文档](../env_setup.md)

### 训练

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
python demo.py --model_path src/human3r_896L.pth --seq_path examples/video.mp4 ...
```

## 文档目录

| 文档 | 内容 |
|------|------|
| [环境配置](../env_setup.md) | Python 环境、依赖安装、RoPE 编译 |
| [训练配置](training.md) | 硬件配置、Batch Size、分布式训练、梯度累积 |
| [模型架构](model.md) | Shot-Aware Adaptation 模块设计 |
| [代码详解](../train_code_explanation.md) | 训练代码流程解析 |
| [TODO](../TODO.md) | 待实现功能清单 |

## 核心特性

### Shot-Aware Adaptation

> **旧版内容备份**：下面三条是早期 StateGate + Residual Adapter 方案描述，当前实现已更新为 ShotTokenGenerator + LoRA Head V1。新设计说明将在下方新增。

处理镜头跳变的轻量微调模块：

- **ShotTokenGenerator**：基于相邻帧差异生成 shot token
- **StateGate**：（计划移除）软性门控状态更新
- **Residual Adapter**：对 base model 输出做微调修正

### 当前实现（LoRA Head V1）

当前训练版本保持原 Human3R recurrent state 行为，不再强制重置到 `S0`，也暂不启用 StateGate。

- **ShotTokenGenerator**：基于相邻帧 decoder image token 差异生成 `q_t`
- **Shot token prompt**：将 `q_t` 拼接到 `[pose, image, human]` token 后进入 decoder cross-attention
- **PoseLoRALayer**：修正 `camera_pose` 的 translation + quaternion
- **HumanLoRALayer**：只修正 `smpl_transl`，不改 shape / rotmat / expression
- **WorldLoRALayer**：对 `pts3d_in_self_view` 和 `pts3d_in_other_view` 做全局 3D shift

`shot_label` 当前不作为显式监督使用。V1 依赖相邻帧特征差异和最终任务 loss 隐式学习 shot-aware correction；后续如需显式检测镜头跳变，可增加 `shot_label` 辅助 BCE loss。

### 训练策略

- 只训练 ~1.3M 参数（0.1%）
- 支持单卡/多卡训练
- bf16 混合精度
- Gradient Checkpointing 节省显存

## 项目结构

```
Movie3R/
├── src/
│   ├── train.py              # 训练入口
│   ├── demo.py               # 推理演示
│   └── dust3r/
│       ├── model.py          # 模型定义
│       ├── shot_adaptation.py # Shot-Aware Adaptation
│       └── datasets/         # 数据集加载
├── config/
│   └── train.yaml            # 训练配置
├── docs/
│   ├── movie3r/              # Movie3R 文档
│   └── train.md              # Human3R 原版训练
└── scripts/
    └── train.sh              # 训练启动脚本
```
