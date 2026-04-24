# Human3R 训练记录

## 实验配置

### 硬件环境
- GPU: NVIDIA H800 (80GB)
- 容器共享内存: /dev/shm 64MB (不足以支持多进程DataLoader)

### 数据集划分
```
train_dataset:
  800 @ AvatarReX_Video(zzr, seed=11) + 800 @ AvatarReX_Video(lbn1, seed=11) + 800 @ AvatarReX_Video(zxc, seed=11)
  + 800 @ AvatarReX_AABB(zzr, seed=11) + 800 @ AvatarReX_AABB(lbn1, seed=11) + 800 @ AvatarReX_AABB(zxc, seed=11)
  = 4800 samples/epoch，Video/AABB 各 50%

val_dataset:（10%）
  100 @ AvatarReX_Video(zzr, seed=22) + 100 @ AvatarReX_Video(lbn1, seed=22) + 100 @ AvatarReX_Video(zxc, seed=22)
  + 100 @ AvatarReX_AABB(zzr, seed=22) + 100 @ AvatarReX_AABB(lbn1, seed=22) + 100 @ AvatarReX_AABB(zxc, seed=22)
  = 600 samples，Video/AABB 各 50%

test_dataset:（10%）
  100 @ AvatarReX_Video(zzr, seed=33) + 100 @ AvatarReX_Video(lbn1, seed=33) + 100 @ AvatarReX_Video(zxc, seed=33)
  + 100 @ AvatarReX_AABB(zzr, seed=33) + 100 @ AvatarReX_AABB(lbn1, seed=33) + 100 @ AvatarReX_AABB(zxc, seed=33)
  = 600 samples，Video/AABB 各 50%
```

**说明**: seed 不同确保 train/val/test 样本不重叠

**原始数据分布**:
| 数据集 | 类型 | 路径 | 样本数 |
|--------|------|------|--------|
| avatarrex_zzr | Video | /workspace/data/avatarrex_zzr_output | 1000 |
| avatarrex_zzr | AABB | /workspace/data/avatarrex_zzr_output | 1000 |
| avatarrex_lbn1 | Video | /workspace/data/avatarrex_lbn1_output | 1000 |
| avatarrex_lbn1 | AABB | /workspace/data/avatarrex_lbn1_output | 1000 |
| avatarrex_zxc | Video | /workspace/data/avatarrex_zxc_output | 1000 |
| avatarrex_zxc | AABB | /workspace/data/avatarrex_zxc_output | 1000 |
| **总计** | | | **6000 samples/epoch** |

### 训练配置
- 模型: ARCroco3DStereo (Human3R 896L)
- 预训练权重: `/workspace/code/Movie3R/src/human3r_896L.pth`
- backbone: dinov2_vitl14
- freeze: none (全量微调)
- num_workers: 0 (单进程，规避/dev/shm限制)
- amp: 混合精度训练
- print_img_freq: 999999 (禁用可视化输出)
- eval_freq: 1 (每epoch评估一次)
- early_stopping_patience: 10 (连续10个epoch不下降则停止)

### Loss组成
```
train_criterion = ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)
               + RGBLoss(MSE)
               + SMPLLoss(L1)
```
包含:
- ConfLoss: 置信度加权3D点回归
- RGBLoss: 图像重建
- SMPLLoss: SMPL参数 (rotmat, transl, shape, j3d, v3d, j2d)
- pose_loss_view2_AABB: AABB跨相机跳变帧的位姿loss (额外添加)

---

## Batch Size 显存测试

### 测试环境
- GPU: H800 80GB
- num_workers: 0
- resolution: 512x288/512x384等 (多尺度)

### 测试结果

| batch_size | 显存使用 | 状态 | 备注 |
|------------|----------|------|------|
| 1 | ~51GB | ✅ OK | 可正常训练 |
| 2 | ~53GB | ✅ OK | 可正常训练 |
| 4 | ~77GB | ❌ OOM | OutOfMemoryError |

**结论: 单卡最大batch_size=2**

---

## 训练结果

### 1 Epoch (batch_size=1) - 2026-04-23

**配置**
- epochs: 1
- batch_size: 1
- steps: 6000
- 训练时间: 4小时44分钟

**Loss曲线**

| Step | Total Loss | 备注 |
|------|-----------|------|
| 0 | 43.90 | 初始 |
| 10 | 13.08 | 快速下降 |
| 20 | 12.52 | |
| 250 | 0.53 | |
| 2900 | ~4.0 | 波动 |
| 5990 | 0.54 | 最终 |

**关键子Loss (最终Step 5990)**
| Loss项 | 值 |
|--------|-----|
| Regr3DPoseBatchList_self_pts3d | 0.029 |
| Regr3DPoseBatchList_pts3d | 0.032-0.085 |
| pose_loss | 0.031 |
| pose_loss_view2_AABB | 0.57 |
| SMPLLoss_j3d | 0.079-0.088 |
| SMPLLoss_rotmat | 0.116 |
| SMPLLoss_shape | 0.049-0.075 |
| SMPLLoss_j2d | 9.7-17.2 |

**Checkpoint保存**
- `checkpoint-final.pth` (4.7GB)
- `checkpoint-last.pth` (11.5GB)

---

### Batch Size=2 训练测试 (未完成1 epoch)

**配置**
- epochs: 1
- batch_size: 2
- steps: 3000

**初步数据**

| Step | Total Loss | 显存 | 每步耗时 |
|------|-----------|------|----------|
| 0 | 32.62 | 46GB | 13s |
| 10 | 13.85 | 53GB | 6.4s |
| 20 | 14.93 | 53GB | 5.4s |

**预计1 epoch时间**: ~7.5小时 (vs batch_size=1的6.7小时，无明显优势)

---

## 后续训练计划

### 推荐配置 (单卡)
- batch_size: 2
- epochs: 20-30
- 预计时间: 20 epochs × 7.5小时 ≈ 150小时 (6天)

### Early Stopping策略
- 监控验证集loss
- 连续5-10个epoch不下降则停止

### 多卡训练 (待解决)
- NCCL allreduce在当前服务器上有问题
- 需联系管理员或换服务器

---

## 已知问题与解决

1. **Dinov2 GitHub网络超时**
   - 解决: 修改`src/mhmr/blocks/dinov2.py`，直接从缓存import

2. **/dev/shm共享内存不足**
   - 解决: num_workers=0 单进程模式

3. **可视化输出导致崩溃**
   - 解决: print_img_freq=999999 禁用可视化

4. **batch_size>2 OOM**
   - 解决: 单卡最大batch_size=2
