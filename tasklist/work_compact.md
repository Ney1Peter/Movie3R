# AvatarReX → Human3R 数据构建与微调

## 1. 项目背景

**目标**: 将 AvatarReX 动作捕捉数据转换为 Human3R 格式，用于训练 Human3R 模型处理镜头跳变场景。

**核心问题**: 原版 Human3R 在镜头连续、相机运动平滑的视频中表现较好，但存在明显镜头切换时模型表现下降。

**方案**: Shot-Aware Adaptation - 在冻结的 CUT3R 基础上增加轻量可学习模块，让模型学会处理镜头跳变。

---

## 2. 数据集处理

### 2.1 原始数据

| 数据集 | 序列数 | 帧数/序列 | 相机数 | 路径 |
|--------|--------|----------|--------|------|
| avatarrex_zzr | 15 | 2001 | 15 | /workspace/data/avatarrex_zzr |
| avatarrex_lbn1 | 16 | 1901 | 15 | /workspace/data/avatarrex_lbn1 |
| avatarrex_zxc | - | - | - | /workspace/data/avatarrex_zxc |

### 2.2 预处理流程

**脚本**: `preprocess_avatarrex_fast.py`
- 多进程并行处理（32 workers）
- 增量处理（已存在则跳过）
- 图像: BGR 格式，2048×1500 → 512×288

**输出格式**:
```
{seq_id}/
  rgb/{frame:08d}.png    # BGR 图像
  cam/{frame:08d}.npz    # pose(4,4) + intrinsics(3,3)
  smpl/{frame:08d}.pkl   # SMPLX 参数
  depth/{frame:08d}.png   # 深度图 (uint16, mm)
  mask/{frame:08d}.png   # 二值 mask
```

**深度图生成**: Depth-Anything-3 (DA3-base)，5 GPU 并行

### 2.3 数据格式验证

| 项目 | 状态 |
|------|------|
| Camera pose | 4×4 c2w 矩阵，与原始标定完全一致 |
| SMPLX 参数 | shape(10) + body_pose(21×3) + hand_pose(15×3)，与原始一致 |
| 深度图 | 2048×1500 uint16 (mm)，范围 0.38m ~ 1.72m |
| Mask | 二值 mask，前景占比约 14.7% |

### 2.4 AvatarReX 数据集类

| 类 | 采样方式 | is_video |
|----|----------|----------|
| `AvatarReX_Video` | 同一相机连续帧 (t, t+1, t+2, t+3) | True |
| `AvatarReX_AABB` | 跨相机跳变 (camA@t, camA@t+1, camB@t+2, camB@t+3) | False |

**shot_label 定义**: `shot_label[i]` 表示 frame i-1 → frame i 是否发生 shot change

| 数据类型 | shot_label | 说明 |
|----------|------------|------|
| Video | [0, 0, 0, 0] | 无跳变 |
| AABB | [0, 0, 1, 0] | frame1→frame2 跳变 |

---

## 3. 模型架构

### 3.1 模型类

**模型**: `ARCroco3DStereo`（继承自 `CroCoNet`）

### 3.2 模型规模 (Human3R 896L)

| 模块 | 参数量 | 占比 |
|------|--------|------|
| backbone (Dinov2) | 304M | 26.1% |
| enc_blocks (ViT) | 302M | 25.9% |
| pose_retriever | 152M | 13.0% |
| downstream_head | 152M | 13.0% |
| dec_blocks | 113M | 9.7% |
| dec_blocks_state | 113M | 9.7% |
| enc_blocks_ray_map | 25M | 2.2% |
| **总计** | **~1.18B** | 100% |

### 3.3 freeze 选项

| freeze 参数 | 冻结内容 | 微调内容 |
|------------|---------|---------|
| `none` | 无 | 全部 (1.18B) |
| `encoder` | enc + backbone | decoder + head (~530M) |
| `encoder_and_decoder_and_head` | enc + dec + head | backbone + mlp_classif/offset |
| `shot_adaptation` | enc + dec + head + S0 | 新增模块 (~1.3M) |

---

## 4. Shot-Aware Adaptation 方案

### 4.1 核心原则

1. **不修改 CUT3R 基模**: encoder/decoder 全部冻结
2. **新增轻量模块**: 只训练新增的 ~1.3M 参数
3. **residual 形式**: LoRA 输出修正量，不直接覆盖原输出

### 4.2 新增模块

| 模块 | 参数量 | 作用 |
|------|--------|------|
| `ShotTokenGenerator` | ~787K | 基于相邻帧差异生成 shot token q_t |
| `StateGate` | ~99K | 控制 state 混合比例 alpha |
| `LoRAPoseHead` | ~198K | 修正相机位姿 (trans+quat, 7D) |
| `LoRAHumanHead` | ~20K | 修正 SMPL 参数 (shape/transl) |
| `LoRAWorldGlobalShift` | ~197K | 全局平移修正点云 |
| **总计** | **~1.3M** | |

### 4.3 数据流

```
F_dec[i], F_dec[i-1] → ShotTokenGenerator → q_t
                                          ↓
                                StateGate → α
                                          ↓
                            S_tilde = α·S_prev + (1-α)·S0
                                          ↓
                      [z, F_t, H_t, q_t] + S_tilde → Decoder
                                          ↓
                                [z', F', H', q']
                                          ↓
                          q_out = tokens[-1:]
                                          ↓
                  LoRA(z', q_out), LoRA(H', q_out), LoRA(F', q_out)
                                          ↓
                                修正后的输出
```

### 4.4 两种模式对比

| 模式 | enable_shot_adaptation=False | enable_shot_adaptation=True |
|------|------------------------------|----------------------------|
| 路径 | 原 Human3R | Shot Adaptation |
| q_t | 不生成 | 预计算后传入 decoder |
| StateGate | 不使用 | alpha 控制 S_tilde |
| LoRA | 不应用 | 修正 pose/human/world |
| 输出 | 等价原 Human3R | 修正后的输出 |

### 4.5 freeze='shot_adaptation' 冻结内容

- **冻结**: encoder / decoder / base heads / S0 (原始 initial state)
- **训练**: ShotTokenGenerator / StateGate / LoRA heads / gamma parameters

---

## 5. 训练配置

### 5.1 训练数据集

| 数据集 | 类型 | 路径 |
|--------|------|------|
| AvatarReX_Video (zzr) | Video | /workspace/data/avatarrex_zzr |
| AvatarReX_Video (lbn1) | Video | /workspace/data/avatarrex_lbn1 |
| AvatarReX_Video (zxc) | Video | /workspace/data/avatarrex_zxc |
| AvatarReX_AABB (zzr) | AABB | /workspace/data/avatarrex_zzr |
| AvatarReX_AABB (lbn1) | AABB | /workspace/data/avatarrex_lbn1 |
| AvatarReX_AABB (zxc) | AABB | /workspace/data/avatarrex_zxc |

### 5.2 数据集划分

| Split | 样本数 | Seed |
|-------|--------|------|
| train | 4800 (800×6) | 11 |
| val | 600 (100×6) | 22 |
| test | 600 (100×6) | 33 |

### 5.3 正式训练参数 (30 epochs)

| 参数 | 值 |
|------|-----|
| batch_size | 2 (per GPU) |
| num GPUs | 4 |
| learning rate | 1e-4 |
| min_lr | 1e-6 |
| warmup_epochs | 5 |
| weight_decay | 0.05 |
| gradient_checkpointing | true |
| amp | true |
| early_stopping_patience | 10 |

### 5.4 训练结果 (30 epochs)

- 训练时长: 20 小时 2 分钟
- Val Loss: 28.52 → 1.31 (降低 95%)
- SMPLLoss_j3d: 0.47m → 0.04m (降低 92%)
- 无过拟合迹象

---

## 6. 已知问题与解决

### 6.1 SMPL 坐标系统错误

**问题**: 早期使用 `smplx_transl[-1] > 0.01` 过滤帧，过滤掉了大量帧（mocap Z 几乎都在 0 附近）

**原因**: `smplx_transl` 存的是 mocap 世界坐标系，不是相机坐标系

**解决**: 过滤前先变换到相机坐标系 `smpl_cam = R_c2w.T @ (smpl_world - t_c2w)`，按 camera_z > -0.5m 过滤

### 6.2 全量微调泛化能力下降

**问题**: freeze='none' 全量微调后，backbone 被微调，模型在 h36.mp4 推理时 SMPL 检测失败（smpl_scores 仅 0.067，低于阈值 0.3）

**原因**: backbone 在 AvatarReX 数据上过拟合，失去对陌生数据的泛化能力

**解决**: 采用 freeze='shot_adaptation' 方案，冻结 backbone，只训练新增轻量模块

---

## 7. 当前进度

| 阶段 | 状态 |
|------|------|
| 数据集预处理与验证 | ✅ 完成 |
| 全量微调验证 (freeze='none') | ✅ 完成 |
| ShotTokenGenerator / StateGate 实现 | ✅ 完成 |
| LoRA Heads 实现 | ✅ 完成 |
| freeze='shot_adaptation' 配置 | ✅ 完成 |
| 数据集 shot_label 添加 | ✅ 完成 |
| Shot Adaptation 训练 | ⏳ 待进行 |

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `src/dust3r/shot_adaptation.py` | Shot-Aware Adaptation 模块 |
| `src/dust3r/model.py` | 模型定义与集成 |
| `src/dust3r/datasets/avatarrex.py` | AvatarReX 数据集类 |
| `src/dust3r/losses.py` | 损失函数 |
| `config/train.yaml` | 训练配置 |
