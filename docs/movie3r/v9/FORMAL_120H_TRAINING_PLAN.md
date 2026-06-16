# V9 Formal 120h Training Plan

这份文档记录下一轮大批量训练准备。当前还没有启动训练。

## 目标

在 V9 implicit human-pose correction 的基础上做更充分训练，数据扩大到 AvatarReX `lbn1 + lbn2 + zzr` 和 THuman `00 + 02`，目标时长约 120 小时。核心仍然比较两组：

| 实验 | 目的 | 配置 |
|---|---|---|
| pose LoRA | 只微调原 Human3R pose head 的 LoRA，检查相机纠正能力 | `config/train_v9_120h_avatarrex_thuman_pose_lora_bs16.yaml` |
| pose + human LoRA | 同时微调 pose head 和 human head 的 LoRA，检查人体 translation 纠正能力 | `config/train_v9_120h_avatarrex_thuman_pose_human_lora_bs16.yaml` |

两组都必须从原版 Human3R 权重开始：

```text
src/human3r_896L.pth
```

不要从任何 V8/V9 小数据、overfit 或 60h/2x checkpoint 继续训练。

## 当前数据状态

已经存在并可用：

| 数据 | 路径 | 大小 |
|---|---|---:|
| AvatarReX lbn1 | `/data/wangzheng/iJCV-CODE/data/Training/lbn1` | 96G |
| AvatarReX lbn2 | `/data/wangzheng/iJCV-CODE/data/Training/lbn2` | 96G |
| AvatarReX zzr | `/data/wangzheng/iJCV-CODE/data/Training/zzr` | 114G |
| THuman00 | `/data/wangzheng/iJCV-CODE/data/Training/thuman00` | 119G |
| THuman02 | `/data/wangzheng/iJCV-CODE/data/Training/thuman02` | 165G |

## Manifest 设计

AvatarReX manifest 已生成：

```text
output/v9_120h_avatarrex_lbn1_lbn2_zzr_48k_manifests
```

规模：

| split | AABB | AAAA | 说明 |
|---|---:|---:|---|
| train | 36000 | 12000 | `lbn1/lbn2/zzr` 各 1/3 |
| val | 3000 | 1000 | 角度桶均衡 |
| test | 3000 | 1000 | 角度桶均衡 |

检查结果：

```text
duplicate clip key: no
train/val frame overlap: 0
train/test frame overlap: 0
val/test frame overlap: 0
AABB angle range: 16.3° - 176.9°
```

THuman manifest 已生成：

```text
output/v9_120h_thuman00_02_32k_manifests
```

计划规模：

| split | AABB | AAAA |
|---|---:|---:|
| train | 24000 | 8000 |
| val | 3000 | 1000 |
| test | 3000 | 1000 |

检查结果：

```text
duplicate clip key: no
train/val frame overlap: 0
train/test frame overlap: 0
val/test frame overlap: 0
AABB angle range: 15.0° - 178.8°
```

生成命令：

```bash
.venv/bin/python scripts/v8_4_build_mixed_aabb_aaaa_manifests.py \
  --training_root /data/wangzheng/iJCV-CODE/data/Training \
  --test_root /data/wangzheng/iJCV-CODE/data/Test/v9_120h_thuman00_02 \
  --output_dir output/v9_120h_thuman00_02_32k_manifests \
  --groups thuman00 thuman02 \
  --train_aabb 24000 --train_aaaa 8000 \
  --val_aabb 3000 --val_aaaa 1000 \
  --test_aabb 3000 --test_aaaa 1000 \
  --min_aabb_angle 15 \
  --overwrite
```

## Split 修正

这次准备时发现旧 manifest split guard 对 AABB/AAAA 混合不够严格。AABB 使用 `start/start+1` 和 `start+2/start+3`，旧的 start guard 可能让 train AABB 的后两帧和 val AAAA 的前两帧重叠。

已修正：

```text
scripts/v8_4_build_mixed_aabb_aaaa_manifests.py
```

现在 split guard 使用 `NUM_VIEWS * 2`，正式 manifest 必须满足 train/val/test frame overlap 全为 0。

## 混合分辨率训练

继续使用 dataset-aware / resolution-aware source batches：

1. AvatarReX 单独组成 batch，保持 `resize_only_16`，不裁剪、不 padding。
2. THuman 单独组成 batch，保持 `resize_only_16`，不裁剪、不 padding。
3. 两个 source batch 分别 forward/backward。
4. 两个 loss 都 backward 后，只做一次 `optimizer.step()`。

训练目标：

```text
L_total = 0.6 * L_AvatarReX + 0.4 * L_THuman
```

保持 0.6/0.4 是为了和 60h/2x 训练可比，同时让更关键的 AvatarReX drift 数据占更高权重。

## 训练日程

基于上一轮 `bs16` 训练日志：

```text
batch size: 16 per source
global batch size: 32
peak memory: about 30GB
step time: about 15.8s
```

充分训练计划：

```text
batch size: 16 per source
train_mixed_epoch_steps: 150
epochs: 200
optimizer steps: 30000
estimated time: about 132 hours, can be stopped early if curves are stable
eval_freq: 10
save_freq: 10
early stopping: disabled
gradient_checkpointing: true
cuda_cache_reserve_mb: 36000
```

`train_mixed_epoch_steps=150` 表示人为定义每个 epoch 有 150 次 optimizer step，不要求完整扫完 manifest。训练充分性主要看总 optimizer steps 和 val/test 指标趋势。

每个 optimizer step 会处理一个 AvatarReX source batch 和一个 THuman source batch：

```text
AvatarReX: 16 clips/step
THuman:    16 clips/step
total:     32 clips/step = 128 frames/step
```

因此整轮训练约等于：

```text
AvatarReX clip samples: 30000 * 16 = 480000
THuman clip samples:    30000 * 16 = 480000
total clip samples:     960000
total frame samples:    3840000
```

相对于当前 manifest：

```text
AvatarReX unique train clips: 48000, about 10.0 passes
THuman unique train clips:    32000, about 15.0 passes
```

## Checkpoint 策略

只保留：

```text
checkpoint-last.pth
checkpoint-best.pth
checkpoint-final.pth
```

`checkpoint-last` 用于中断恢复，`checkpoint-best` 用验证指标选择，`checkpoint-final` 用完整训练后的结果。不要保存每个 epoch 的大权重。

当前 `save_freq=10`，也就是每 10 epoch 保存一次 `checkpoint-last`。在 `150 steps/epoch` 下，每次间隔保存约为 1500 optimizer steps，按上一轮速度约 6.5 小时一次。`checkpoint-last` 会覆盖旧 last，避免磁盘被中间权重填满。

## GPU 显存策略

训练过程中实际 activation 峰值会在不同 source batch、val/test 和日志阶段之间波动。为了避免服务器上别人误把任务放到同一张 GPU 后在我们峰值阶段造成 OOM，120h 配置开启了 PyTorch CUDA cache 预留：

```text
cuda_cache_reserve_mb: 36000
cuda_cache_reserve_safety_mb: 8192
cuda_cache_reserve_chunk_mb: 512
```

含义：

- 每个 mixed training epoch 开始时，训练进程会把 PyTorch CUDA caching allocator 预热到约 36GB。
- 预热 tensor 会立刻删除，因此这部分显存不是死占用，后续 forward/backward 可以复用。
- 这不会改变模型输入、loss、梯度、optimizer step 或训练结果，只改变本进程向 CUDA 申请显存的时机。
- 如果 GPU 总显存不够，代码会保留 `cuda_cache_reserve_safety_mb` 的余量，并自动降低可申请上限。
- `train_steps.jsonl` 现在同时记录 `max_mem_mb` 和 `reserved_mem_mb`，前者是真实计算峰值，后者是 PyTorch 当前缓存占用。

已经测试过 `gradient_checkpointing=false + batch_size=16`，在 L20 上会 OOM：

```text
PyTorch allocated: about 43.0GB
extra request:     about 1.44GB
result:            CUDA out of memory
```

因此正式 120h 配置保持：

```text
gradient_checkpointing: true
batch_size: 16
```

`gradient_checkpointing=true + batch_size=16` 的 10-step smoke test 可以跑通：

```text
max_mem_mb:      about 35870MB
reserved_mem_mb: about 31.4GB - 34.1GB
step time:       about 15.5s - 20.4s after warmup
```

注意：`nvidia-smi` 外部采样不一定一直显示 35GB，训练中看到过约 27GB 和 34GB 两种状态；但训练内部真实峰值稳定在约 35.9GB。

## 训练前检查

正式启动前必须完成：

- THuman00/02 的 SMPL/camera 坐标系投影验证通过。
- THuman manifest 已生成，metadata 中 frame overlap 全为 0。
- 两个 120h config 可以 Hydra 解析。
- `src/human3r_896L.pth` 存在。
- GPU6/GPU7 或指定 GPU 空闲。
- 没有旧 viewer、临时推理、smoke train 进程占 GPU。
- 输出目录 `output/v9_120h_mixed/` 空间足够。
- 正式 benchmark 仍使用 zxc held-out、AvatarReX in-domain、THuman in-domain，并包含 camera + human + MPJPE/PA-MPJPE/PVE 指标。

## 启动命令

已经准备好安全启动脚本：

```bash
bash scripts/training/run_v9_120h_dual_lora.sh
```

默认只打印命令，不启动训练。确认 GPU6/GPU7 空闲后，才使用：

```bash
bash scripts/training/run_v9_120h_dual_lora.sh --start
```

脚本会启动两个 tmux session：

```text
v9_120h_pose_lora_gpu6
v9_120h_pose_human_lora_gpu7
```

确认所有检查通过后再启动：

```bash
MPLCONFIGDIR=/data/wangzheng/iJCV-CODE/Movie3R/output/tmp/mpl \
CUDA_VISIBLE_DEVICES=6 \
.venv/bin/python src/train.py --config-name train_v9_120h_avatarrex_thuman_pose_lora_bs16
```

```bash
MPLCONFIGDIR=/data/wangzheng/iJCV-CODE/Movie3R/output/tmp/mpl \
CUDA_VISIBLE_DEVICES=7 \
.venv/bin/python src/train.py --config-name train_v9_120h_avatarrex_thuman_pose_human_lora_bs16
```

现在不要直接启动，先做 dataloader sanity check 和 GPU 空闲检查。
