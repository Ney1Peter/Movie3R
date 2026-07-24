# V9 Formal 60h Training Plan

这份文档记录下一轮正式训练方案。当前只是准备阶段，训练还没有开始。

## 目标

用 AvatarReX 和 THuman 做混合训练，验证 V9 implicit human-pose correction 在更多数据上是否真的学到前馈纠正能力，而不是只在单 clip 或小数据上过拟合。

核心对比只跑两组：

| 实验 | 目的 | 配置 |
|---|---|---|
| pose LoRA | 只微调 pose head 的 LoRA，检查相机纠正收益 | `config/train_v9_mixed_avatarrex_thuman_60h_pose_lora_bs10.yaml` |
| pose + human LoRA | 同时微调 pose head 和 human head 的 LoRA，检查人体纠正收益 | `config/train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10.yaml` |

两组都从原版 Human3R 权重开始：

```text
src/human3r_896L.pth
```

不要从之前 V8/V9 的实验 checkpoint 继续训练，避免把小数据过拟合结果带入正式实验。

## 数据划分

训练使用两个数据源，但每个数据源保持自己的 resize 后比例，不做 padding，不裁剪到统一正方形。

| 数据源 | 训练 | 验证 | 测试 | 说明 |
|---|---:|---:|---:|---|
| AvatarReX lbn1 + zzr | 8000 AABB + 2000 AAAA | 200 AABB + 50 AAAA | 400 AABB + 100 AAAA | in-domain |
| THuman00 + THuman02 | 8000 AABB + 2000 AAAA | 200 AABB + 50 AAAA | 400 AABB + 100 AAAA | in-domain |
| AvatarReX zxc | 0 | 0 | 1000 AABB + 250 AAAA | held-out test only |

Manifest 固定在：

```text
output/v9_60h_avatarrex_lbn1_zzr_10k_manifests
output/v9_60h_thuman00_02_10k_manifests
output/v9_zxc_heldout_test_manifests
```

已经检查过：

- train / val / test 没有重复 clip key。
- train / val / test 没有 frame overlap。
- zxc 完全不参与训练和验证，只做最终泛化测试。
- AABB 和 AAAA 都保留，AABB 学习跳变纠正，AAAA 约束正常连续帧不要过度修正。

## 混合分辨率训练

AvatarReX 和 THuman 的 resize-only 尺寸不同：

```text
AvatarReX: 约 368 x 512
THuman:    约 512 x 448
```

同一个 PyTorch batch 不能直接 stack 不同 H/W 的图片。正式训练采用 dataset-aware gradient accumulation：

1. AvatarReX 单独组成一个 batch，保持 `resize_only_16`。
2. THuman 单独组成一个 batch，保持 `resize_only_16`。
3. 两个 batch 分别 forward/backward。
4. 两个 loss 都 backward 后，只做一次 `optimizer.step()`。
5. 两个 backward 中间不 `optimizer.zero_grad()`。

优化目标等价于：

```text
L_total = 0.6 * L_AvatarReX + 0.4 * L_THuman
```

当前权重设置为 AvatarReX 0.6、THuman 0.4，因为 AvatarReX 更贴近我们关心的 drift 场景；THuman 用来补充纹理丰富、Human3R raw 本来较稳的正常/轻漂移样本。

## 训练日程

当前计划：

```text
batch size: 10 per dataset source
train_mixed_epoch_steps: 100
epochs: 72
optimizer steps: 7200
eval_freq: 6
save_freq: 6
early stopping: disabled
gradient_checkpointing: true
```

预计总时长约 60 小时。实际速度以 GPU6/GPU7 的实时日志为准。

`train_mixed_epoch_steps=100` 的含义是每个 epoch 人为定义为 100 次 optimizer step，而不是把 10k manifest 完整扫一遍才算一个 epoch。这样做的目的是固定训练总步数和评估频率，让训练时长更可控。

## Checkpoint 策略

正式训练只需要保留：

```text
checkpoint-last.pth
checkpoint-best.pth
checkpoint-final.pth
```

`checkpoint-last` 用于中断恢复，`checkpoint-best` 用验证集选择，`checkpoint-final` 用最终充分训练后的结果。中间每个 epoch 的 checkpoint 不保留，避免磁盘被权重占满。

## 训练指标

训练过程中重点看这些数值：

| 指标 | 含义 | 期望 |
|---|---|---|
| `loss` / `v82_pose_relation_loss` | 总训练目标 | 稳定下降，后期趋稳 |
| `v82_trans_err` | corrected camera translation error | AABB 明显低于 raw |
| `v82_rot_err_deg` | corrected camera rotation error | 不恶化，最好下降 |
| `v82_raw_trans_err` / `v82_raw_rot_err_deg` | 原始 Human3R raw 误差 | 作为对照，不会被训练改变 |
| `v82_gate_mean` | learned gate 平均值 | AABB 高于 AAAA |
| `v82_human_trans_err` | corrected SMPL translation error | pose+human LoRA 应更好 |
| `v82_raw_human_trans_err` | raw SMPL translation error | 作为人体纠正对照 |
| `v82_delta_norm` | pose correction delta 大小 | 不应无限变大 |
| `v82_human_latent_delta_norm` | human latent correction delta 大小 | 不应无限变大 |
| `v82_pose_head_lora_l2` | pose LoRA 残差规模 | 有学习但不过大 |
| `v82_human_head_lora_l2` | human LoRA 残差规模 | pose+human LoRA 中观察 |

## 测试指标

训练完成后，不能只看 viewer。正式 benchmark 需要输出：

| 指标 | 单位 | 含义 |
|---|---:|---|
| camera translation error | m | corrected camera 到 GT camera 的平移误差 |
| camera rotation error | degree | corrected camera 到 GT camera 的旋转误差 |
| raw camera error | m / degree | 原版 Human3R raw camera 对照 |
| gate mean | 0-1 | 模型判断是否需要纠正的平均强度 |
| human translation error | m | corrected SMPL translation 到 GT 的误差 |
| raw human translation error | m | raw SMPL translation 对照 |
| Metric-MPJPE | mm | 不做中心对齐的 3D joint 误差 |
| MPJPE | mm | 按 pelvis 中心对齐后的 3D joint 误差，接近 Human3R 论文里的 MPJPE |
| PA-MPJPE | mm | Procrustes 对齐后的 3D joint 误差 |
| RootError | mm | pelvis/root 平移误差 |
| Metric-PVE / PVE / PA-PVE | mm | mesh vertex 误差，作为补充 |

注意：MPJPE/PA-MPJPE 是评估指标，不参与当前训练反传。训练 loss 仍以 camera pose、human translation、gate/drift、delta regularization 和 LoRA regularization 为主。

## 正式训练命令

训练前先确认 GPU6/GPU7 空闲，再分别启动：

```bash
MPLCONFIGDIR=/data/wangzheng/iJCV-CODE/Movie3R/output/tmp/mpl \
CUDA_VISIBLE_DEVICES=6 \
.venv/bin/python src/train.py --config-name train_v9_mixed_avatarrex_thuman_60h_pose_lora_bs10
```

```bash
MPLCONFIGDIR=/data/wangzheng/iJCV-CODE/Movie3R/output/tmp/mpl \
CUDA_VISIBLE_DEVICES=7 \
.venv/bin/python src/train.py --config-name train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10
```

如果 GPU 被占用，不要抢占别人的进程；等待空闲或改到用户确认的 GPU。

## Benchmark 计划

每个 checkpoint 至少评估三组：

```text
output/v9_60h_benchmarks/avatarrex_in_domain
output/v9_60h_benchmarks/thuman_in_domain
output/v9_60h_benchmarks/zxc_heldout
```

最终汇报时至少给出：

- AvatarReX in-domain AABB / AAAA 表格。
- THuman in-domain AABB / AAAA 表格。
- zxc held-out AABB / AAAA 表格。
- 每组各挑 2-3 个 viewer case：大角度 AABB、小角度 AABB、AAAA。
- viewer 中必须同时有 GT、original Human3R raw、corrected，且 raw 必须来自原版 Human3R demo 输出或经过验证的同坐标输出。

## 训练前检查清单

正式启动前必须检查：

- 两个 formal config 都能通过 Hydra 解析。
- `src/human3r_896L.pth` 存在。
- manifest 路径存在，且 metadata 中 split 数量正确。
- `output/v9_mixed_60h/` 有足够空间。
- 没有旧 viewer / smoke train / 临时 benchmark 进程占 GPU。
- GPU6/GPU7 显存足够。
- `scripts/v8_4_eval_pose_benchmark.py` 能输出 camera + human + MPJPE 指标。
- AvatarReX raw calibration root 包含 `lbn1`、`zzr`、`zxc`。
- THuman 的 GT SMPL/camera 坐标系按已验证版本，不再使用错误投影方式。
- 正式训练期间不要写大文件到 `/tmp`，临时文件使用 `output/tmp`。

## 当前不做的事

- 不启用 ASIT 数据集。ASIT 预处理可以保留，但正式训练先只用 AvatarReX + THuman。
- 不做全量 pose head / human head 解冻，只做 LoRA。
- 不做早停。之前早停逻辑容易误停，这次先完整跑完计划步数。
- 不用 padding 到 512 x 512。灰边可能污染训练分布。
- 不从任何小数据过拟合 checkpoint 继续训练。
