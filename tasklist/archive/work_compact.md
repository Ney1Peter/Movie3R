# AvatarReX → Human3R 数据构建与微调

## 1. 项目背景

**目标**: 将 AvatarReX 动作捕捉数据转换为 Human3R 格式，用于训练 Human3R 模型处理镜头跳变场景。

**核心问题**: 原版 Human3R 在镜头连续、相机运动平滑的视频中表现较好，但存在明显镜头切换时模型表现下降。

**方案**: Shot-Aware Adaptation - 在冻结的 CUT3R 基础上增加轻量可学习模块，让模型学会处理镜头跳变。

**2026/05/09 最新状态**：V4 验证了 pose-only ShotToken 比 V2 安全，但 decoder 后单次修正仍偏后处理，且 translation y/z 容易引入额外错位。下一步规划 V5.1：在每层 decoder 后只让 pose token 和 shot token 做 attention，并同步增加 `L_boundary_abs`、`L_jump_rel`、`L_anchor`；若 V5.1 失败，再进入 V5.2 masked decoder 方案。

**2026/05/13 最新状态**：V6 方向调整为 local scene AnchorToken。RICH AABB 实验证明 XFeat semi-dense + official mesh anchors 能映射回 Human3R encoder patch token；affine coarse re-anchor 明显优于简单 mean translation；AnchorToken leave-one-out 验证显示 `global affine + local AnchorToken residual` 在 anchor 数充足时优于纯 affine。Top-K 验证进一步显示推理时不需要保留所有 anchors，8-16 个高质量 / 空间分散 AnchorTokens 通常已能提供有效 residual correction。当前建议不改 encoder、不让 anchor 进入完整 decoder sequence，先把 AnchorToken 作为 pose/camera path 的受控 re-anchor evidence。

**2026/05/13 补充**：新增 AnchorToken specificity / negative-control 验证。strong samples 中 correct AnchorToken 优于 affine-only；shuffled value 和 wrong-boundary token 会退化，说明 token 携带的是具体 local residual correction evidence，而不是泛泛 shot label。已生成 `BBQ_001_guitar` high-overlap offline cache：185/185 samples 成功，保存在 `/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/`。

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
| `ShotTokenGenerator` | ~788K | 基于相邻帧差异生成 shot token q_t |
| `PoseLoRALayer` | ~99K (rank=64) | 修正相机位姿 (trans+quat, 7D) |
| `HumanLoRALayer` | ~98K (rank=64) | 只修正 SMPL 平移 |
| `WorldLoRALayer` | ~98K (rank=64) | 全局平移修正点云 |
| **总计** | **~1.08M** | 当前 LoRA64 配置 |

### 4.3 数据流

```
F_dec[i], F_dec[i-1] → ShotTokenGenerator → q_t
                                          ↓
                      [z, F_t, H_t, q_t] + 原 Human3R recurrent state → Decoder
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
| StateGate | 不使用 | 不使用，已移除 |
| LoRA | 不应用 | 修正 pose/human/world |
| 输出 | 等价原 Human3R | 修正后的输出 |

### 4.5 freeze='shot_adaptation' 冻结内容

- **冻结**: encoder / decoder / base heads / 原 Human3R recurrent state 相关参数
- **训练**: ShotTokenGenerator / LoRA heads / gamma parameters

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

- LoRA64 正式训练目录: `experiments/formal_training-4gpu-lora-64`
- train loss 和 AvatarReX val/test loss 下降
- `checkpoint-best.pth` 推理 demo 失败，不能作为可用模型
- 消融结果：关闭 `enable_shot_adaptation` 后恢复 base Human3R 尺度，打开后 camera/pointmap/SMPL 尺度崩坏

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
| Shot Adaptation 训练 | ⚠️ LoRA64 已完成但推理失败 |
| Shot token 质量验证 | ✅ 已验证输入特征可区分跳变，问题主要在注入方式 |
| V4 pose-only alignment | ✅ 已验证安全但偏后处理，B 段 y/z 仍有错位 |
| V5.1 layerwise pose-only attention | ⏳ 下一步实现 |
| V5.2 masked decoder | ⏸️ V5.1 失败后再考虑 |
| V6 AnchorToken specificity 验证 | ✅ correct token 优于 affine，负例退化 |
| Guitar offline AnchorToken cache | ✅ high-overlap 185 samples 已生成 |

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `src/dust3r/shot_adaptation.py` | Shot-Aware Adaptation 模块 |
| `src/dust3r/model.py` | 模型定义与集成 |
| `src/dust3r/datasets/avatarrex.py` | AvatarReX 数据集类 |
| `src/dust3r/losses.py` | 损失函数 |
| `config/train.yaml` | 训练配置 |
| `docs/movie3r/shot_token_v5_plan.md` | V5.1/V5.2 规划文档 |
