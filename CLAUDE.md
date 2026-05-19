# Movie3R 项目指南

## 项目概述

Movie3R 是基于 Human3R 的扩展，针对**多镜头电影级人体重建**场景优化，主要研究**镜头跳变（shot change）**带来的时序不连续问题。

当前项目已从 V2-V6 的 ShotToken / background AnchorToken 路线切换到 **V7 调研阶段**。近期测试显示，Human3R 的明显偏移主要出现在低纹理、弱背景特征、简单场景中；RICH / AvatarReX 等纹理丰富数据上原版 Human3R 通常较稳定。

## 关键文件

| 文件 | 用途 |
|------|------|
| `src/train.py` | 训练入口 |
| `src/demo.py` | 推理演示 |
| `src/dust3r/model.py` | 模型定义（ARCroco3DStereo） |
| `src/dust3r/shot_adaptation.py` | 历史 Shot-Aware Adaptation 模块 |
| `src/dust3r/datasets/avatarrex.py` | AvatarReX 数据集加载 |
| `config/train.yaml` | 训练配置 |
| `train.sh` | 训练启动脚本 |

## 训练相关

### 启动训练
```bash
cd src
./train.sh [num_gpus] [epochs] [batch_size]
```

### 关键配置
- `num_views: 4` — 每个 sample 包含 4 个视角
- `batch_size` — 每卡 batch size（不是全局）
- `accum_iter` — 梯度累积步数
- `gradient_checkpointing: true` — 节省显存
- `amp: 1` — bf16 混合精度

### 分布式训练
- 使用 `torchrun` + `Accelerate`（封装 DDP）
- 不使用 FSDP 或 DeepSpeed（当前规模不需要）

## 模型架构

### 当前阶段

V7 尚未确定新的模型结构。现有代码仍保留 V2-V6 历史实验路径，包括 ShotToken、LoRA、AnchorPoseAdapter、AnchorToken decoder 和 pose-token adapter 等实现，但这些不再代表当前主线。

V2-V6 历史文档见：

```text
docs/movie3r/archive_v2_v6/
```

## 文档位置

| 文档 | 路径 |
|------|------|
| Movie3R 概览 | `docs/movie3r/README.md` |
| 当前调研情况 | `docs/movie3r/current_research_context.md` |
| V7 入口 | `docs/movie3r/v7/README.md` |
| V2-V6 历史归档 | `docs/movie3r/archive_v2_v6/README.md` |
| 训练代码详解 | `docs/train_code_explanation.md` |
| 待办事项 | `tasklist/TODO.md` |
| 环境配置 | `docs/env_setup.md` |

## 代码修改规范

### 核心原则：永远不删除代码，只注释备份

**重要**：修改代码时，**永远不要直接删除旧代码**。将旧代码用 `**========**` 注释框起来保留，便于回退和审查。

### 保留历史的修改格式
```python
# **========== 原始代码 ==========**
# def old_function():
#     old_code_here
#     pass

# **========== 新代码 ==========**
def new_function():
    new_code_here
    pass
# **========== 结束 ==========**
```

### 修改流程
1. 先用 `**========**` 注释框住要修改的原始代码
2. 在下方编写新代码
3. commit 时说明："保留原始代码注释"
4. 确认新代码稳定后，再考虑是否删除注释掉的旧代码

### 分步提交流程（重要）
1. 第一版 commit：只做原代码注释备份，不新增替代实现，commit message 标明“注释保留原始代码”
2. 第二版 commit：在注释块下方新增实现，commit message 标明“新增实现”
3. 多个相关 commit 完成后再统一 push，便于回退和审查
4. 除非用户明确要求，不要把注释备份和新实现压成一个 commit
5. 除非用户明确要求，不要直接 push

### Commit 规范
```bash
git commit -m "refactor: 移除 StateGate 模块（保留原始代码注释）"
git commit -m "refactor: 将 ResidualAdapter 改为 LoRA（保留原始代码注释）"
```

分开 commit 后再一起 PR，方便追溯历史。

## 参数更新流程

```
with accelerator.accumulate(model)
    │
    ├── 控制"何时"更新 ← Accelerate
    ▼
loss_scaler(loss, optimizer, ...)
    │
    ├── backward()        ← NativeScaler (bf16 scaling)
    ├── clip_grad_norm_() ← NativeScaler (梯度裁剪)
    │
    ▼
optimizer.step()          ← AdamW (实际更新参数)
    │
    ▼
optimizer.zero_grad()
```

- `accelerator.accumulate()`：控制何时更新（梯度累积）
- `NativeScaler`：执行 backward + 梯度裁剪
- `AdamW`：执行 `step()` 实际更新参数

## 常见问题

1. **显存不足**：减小 batch_size 或启用 gradient_checkpointing
2. **多卡训练卡住**：检查 NCCL 版本，尝试 `GLOO_SOCKET_IFNAME=lo`
3. **/dev/shm 不足**：设置 `num_workers=0`
