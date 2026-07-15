# V10 最小验证：BEDLAM/21 Streaming Segment Integrator

日期：2026-07-14

## 1. 验证目标

这次只验证 V10 中最核心的一件事：

> 已知 shot boundary 的情况下，能否用一个严格流式的模块，把新 segment 的局部 Human3R 输出接回历史 global state？

这次不验证 detector。boundary 先使用 oracle，也就是直接告诉模型哪些帧是新段开始。

严格流式约束：

- 当前帧只使用历史输出和当前帧输出；
- 不看未来帧；
- 不跑完整段以后再回头优化；
- 新 segment 的第一帧预测一个 segment-to-global transform；
- 这个 transform 被缓存，本 segment 后续帧全部复用；
- 后续帧不会每帧重新预测大范围对齐。

## 2. 当前模型架构

当前最小模型不是改 Human3R encoder/decoder 主干，而是在 Human3R 输出之后加一个 streaming segment integrator。

整体流程：

```text
单目视频帧
  ↓
原版 Human3R 逐帧/流式推理
  ↓
得到每帧 camera、SMPL-X root、depth/conf/color
  ↓
oracle detector 给出 segment boundary
  ↓
segment 内保持 Human3R 自己的局部连续重建
  ↓
新 segment 第一帧进入 V10 integrator
  ↓
预测 segment-to-global transform
  ↓
本 segment 后续帧复用同一个 transform
  ↓
输出连续 world-gauge camera + human
```

当前重点训练两个模块：

### 2.1 Direct Coarse Integrator

模块名：

```text
history_current_integrator
```

作用：

```text
看历史 global state + 当前 local boundary frame
直接预测一个粗 SE3:
R_direct, t_direct
```

它不是监督显式算出来的 SE3，而是监督应用 SE3 后的输出是否对齐 target。

损失：

```text
8.0 * root_translation_loss
+ 1.5 * root_rotation_loss
+ 2.0 * camera_translation_loss
+ 0.5 * camera_rotation_loss
```

### 2.2 Direct Residual Integrator

模块名：

```text
history_direct_residual_integrator
```

作用：

```text
先使用 history_current_integrator 得到 coarse 对齐
  ↓
把当前 local boundary frame 变换到 coarse global
  ↓
residual head 再预测一个小 SE3:
R_residual, t_residual
  ↓
最终 transform:
R_final = R_residual @ R_direct
t_final = R_residual @ t_direct + t_residual
```

residual 不是从零学习大偏移，而是在 coarse 对齐之后学习小修正。

当前 residual 范围：

```text
residual_max_rot_deg = 25
residual_max_trans = 0.8
```

训练时冻结 direct head，只训练 residual head。

## 3. 为什么暂时不用手工 explicit SE3

之前手工 SE3 粗对齐会出现上下颠倒、roll/pitch 错误等问题。

原因是人体 root 或少量人体点存在歧义：

- 人体近似对称；
- 多人顺序可能有噪声；
- Human3R 输出本身有局部坐标误差；
- 只用人体点很难稳定判断世界 up、地面方向和朝向。

所以当前主线先采用：

```text
可学习 direct 粗对齐 + 可学习 residual 细修
```

手工 explicit SE3 暂时保留为对照方法，不作为主线。

## 4. 数据处理

### 4.1 原始数据

使用：

```text
/data/wangzheng/iJCV-CODE/data/BEDLAM/21
```

已过滤后的有效帧：

```text
0000, 0005, 0010, ..., 0140
```

共 29 帧，每帧 4 个人。

对应 metadata：

```text
config/manifests/bedlam_seq000021_good_6fps/metadata.json
```

### 4.2 Human3R 输出域 cache

主训练域使用 Human3R saved-output，而不是直接用 BEDLAM GT。

原因：

- 推理时 integrator 的输入来自 Human3R 输出；
- Human3R 的 camera、SMPL root、depth、尺度和 GT 不一定完全同分布；
- 直接用 GT 训练可能产生 domain gap；
- Human3R saved-output 同时包含 depth/conf/color，可以后续提取点云背景 cue。

Human3R cache 格式：

```text
camera/{frame:06d}.npz    # pose, intrinsics
smpl/{frame:06d}.npz      # rotvec, transl, shape, expression, msk
depth/{frame:06d}.npy
conf/{frame:06d}.npy
color/{frame:06d}.png
```

最小训练当前只使用：

```text
camera pose
SMPL-X root rot/transl
```

点云暂时不进入训练。

### 4.3 点云背景 cue

点云信息重要，但这次最小验证先不接入训练，避免变量过多。

当前建议是先从 Human3R saved-output 中提取 compact scene cue：

```text
background centroid
background scale
PCA axes / eigenvalues
camera-to-scene vector
human-to-scene vector
```

后续 V10.1 再做：

```text
history-current direct + residual + compact scene cue
```

这次只做不带点云的核心验证。

## 5. Synthetic Segment Perturbation

训练时不是拿真实 shot 数据直接训练，而是在一个连续 Human3R 输出上人为构造 local reset。

步骤：

```text
Human3R 连续输出
  ↓
作为 pseudo-global target
  ↓
按 boundary 切段，例如 0, 10, 20
  ↓
对后续 segment 加随机 SE3
  ↓
得到 local reset 风格输入
  ↓
训练 integrator 把 local segment 接回 pseudo-global target
```

默认扰动：

```text
perturb_rot_deg = 120
perturb_trans = 2.5
global_rot_deg = 180
global_trans = 5.0
```

`global_rot/trans` 会先给整条轨迹加随机世界 gauge，防止模型记住固定 BEDLAM 坐标。

## 6. 训练格式

训练样本不是逐帧全序列，而是每个 synthetic episode 中的 boundary item。

每个 item 包含：

```text
local_root_t / local_root_R
local_cam_t / local_cam_R
history global root/cam
history velocity
expected next root/cam
target_root_t / target_root_R
target_cam_t / target_cam_R
```

核心 feature：

```text
feature_history
```

包含：

- 当前 local 人体 root；
- 当前 local camera；
- 历史 global 人体 root；
- 历史 global camera；
- 历史 velocity；
- expected next state；
- 多人 pairwise root distance。

residual feature：

```text
feature_history_direct_residual
```

包含：

- `feature_history`；
- direct head 预测的 coarse SE3；
- coarse 对齐后的 root；
- coarse 对齐后的 camera。

## 7. 最小验证命令

### 7.1 从 BEDLAM/21 原图重建 Human3R cache

输入原图目录：

```text
/data/wangzheng/iJCV-CODE/data/BEDLAM/21/images/seq_000021
```

取 29 帧：

```text
max_frames = 141
subsample = 5
```

命令：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python scripts/run_human3r_save_output.py \
  --model_path src/human3r_896L.pth \
  --seq_path /data/wangzheng/iJCV-CODE/data/BEDLAM/21/images/seq_000021 \
  --output_dir output/v10_bedlam21_minimal_validation/original_human3r_demo_fresh \
  --device cuda \
  --max_frames 141 \
  --subsample 5 \
  --strict_original_human3r \
  --overwrite
```

### 7.2 训练 Human3R 输出域 integrator

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/v10_bedlam_motion_integrator_probe.py \
  --trajectory_source human3r_saved \
  --human3r_output_dir output/v10_bedlam21_minimal_validation/original_human3r_demo_fresh \
  --num_frames 29 \
  --max_people 4 \
  --output_dir output/v10_bedlam21_minimal_validation/human3r_domain_integrator_fresh \
  --segment_boundaries 0 10 20 \
  --train_episodes 512 \
  --val_episodes 128 \
  --steps 2500 \
  --batch_size 128 \
  --device cuda
```

### 7.3 导出 Human3R viewer payload

```bash
.venv/bin/python scripts/v10_apply_integrator_to_human3r_saved_output.py \
  --run_dir output/v10_bedlam21_minimal_validation/human3r_domain_integrator_fresh \
  --input_dir output/v10_bedlam21_minimal_validation/original_human3r_demo_fresh \
  --output_dir output/v10_bedlam21_minimal_validation/viewer_payload_fresh \
  --num_frames 29 \
  --max_people 4 \
  --segment_boundaries 0 10 20 \
  --device cpu \
  --overwrite
```

### 7.4 GT 域 sanity check

这个不是主训练域，只用来确认同一个 integrator 在干净 BEDLAM GT 上也能学会。

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/v10_bedlam_motion_integrator_probe.py \
  --trajectory_source bedlam_gt \
  --manifest config/manifests/bedlam_seq000021_good_6fps/metadata.json \
  --output_dir output/v10_bedlam21_minimal_validation/bedlam_gt_integrator \
  --segment_boundaries 0 10 20 \
  --train_episodes 512 \
  --val_episodes 128 \
  --steps 2500 \
  --batch_size 128 \
  --device cuda
```

## 8. 判断标准

最小验证通过需要看到：

1. `history_current_integrator` 明显优于 `current_only_mlp`。
2. `history_direct_residual_integrator` 优于 `history_current_integrator`。
3. `raw_perturbed` 明显最差。
4. `explicit_se3_residual_integrator` 不作为主线，只作为参考。
5. Human3R viewer 中，`history_direct_residual_integrator` 的 camera/human 接近 `target_original_human3r`。

当前核心判断：

```text
如果 Human3R 输出域验证成立，说明训练一个 streaming segment integrator 是合理的。
如果 GT 域成立但 Human3R 输出域不成立，说明 domain gap 是主要问题。
如果 Human3R 输出域成立，点云 cue 可以作为下一步增强，而不是当前最小验证的必要条件。
```

## 9. 本次实际执行结果

### 9.1 Human3R cache

本轮从 BEDLAM/21 原图重新生成 Human3R cache：

```text
output/v10_bedlam21_minimal_validation/original_human3r_demo_fresh
```

实际读取的 29 帧为：

```text
0000, 0005, 0010, ..., 0140
```

原图尺寸和 Human3R 输入尺寸：

```text
1280x720 -> 512x288
```

输出目录包含 29 帧：

```text
camera / smpl / depth / conf / color
```

后续 integrator 训练、评估和导出均基于这个 fresh cache 重新跑，没有复用旧权重。

### 9.2 Human3R 输出域结果

训练输出：

```text
output/v10_bedlam21_minimal_validation/human3r_domain_integrator_fresh
```

指标如下，越低越好：

| Variant | Root Trans | Root Rot | Cam Trans | Cam Rot | Boundary Jump | Velocity | Non-boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_perturbed | 5.1911 | 37.71 | 2.8574 | 37.71 | 9.4639 | 0.8508 | 0.1883 |
| fixed_explicit_se3 | 0.9353 | 68.19 | 7.3069 | 68.19 | 2.0954 | 0.5811 | 0.4646 |
| current_only_mlp | 5.2740 | 39.38 | 3.0868 | 39.38 | 8.1762 | 0.7682 | 0.1984 |
| history_current_integrator | 1.3178 | 21.31 | 1.3727 | 21.31 | 1.8060 | 0.2140 | 0.0916 |
| history_direct_residual_integrator | 1.2594 | 19.63 | 1.2185 | 19.63 | 1.7096 | 0.2000 | 0.0839 |
| explicit_se3_residual_integrator | 2.7310 | 71.04 | 6.3876 | 71.04 | 4.2541 | 0.7427 | 0.4726 |
| oracle_se3_upper | 0.0000 | 0.08 | 0.0000 | 0.08 | 0.0000 | 0.0000 | 0.0000 |

结论：

1. `current_only_mlp` 基本无效，说明只看当前 local frame 不够。
2. `history_current_integrator` 明显降低 boundary jump 和 root/camera 误差，说明历史 global state 是必要信息。
3. `history_direct_residual_integrator` 继续优于 `history_current_integrator`，说明“先粗接，再小修”有效。
4. `fixed_explicit_se3` 和 `explicit_se3_residual_integrator` 在 Human3R 输出域失败，尤其 camera/rotation 很差，说明手工 explicit SE3 不能直接作为主线。

### 9.3 BEDLAM GT 域 sanity check

训练输出：

```text
output/v10_bedlam21_minimal_validation/bedlam_gt_integrator
```

指标如下，越低越好：

| Variant | Root Trans | Root Rot | Cam Trans | Cam Rot | Boundary Jump | Velocity | Non-boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_perturbed | 2.8730 | 37.71 | 5.1981 | 37.71 | 5.0944 | 0.3671 | 0.0035 |
| fixed_explicit_se3 | 0.0063 | 3.47 | 0.5706 | 3.47 | 0.0062 | 0.0008 | 0.0004 |
| current_only_mlp | 2.8132 | 40.70 | 5.4823 | 40.70 | 4.1863 | 0.3025 | 0.0038 |
| history_current_integrator | 0.8696 | 19.52 | 1.5467 | 19.52 | 1.1079 | 0.0808 | 0.0018 |
| history_direct_residual_integrator | 0.7997 | 19.21 | 1.4589 | 19.21 | 1.0341 | 0.0755 | 0.0018 |
| explicit_se3_residual_integrator | 0.0226 | 0.36 | 0.0493 | 0.36 | 0.0217 | 0.0016 | 0.0000 |
| oracle_se3_upper | 0.0000 | 0.08 | 0.0000 | 0.08 | 0.0000 | 0.0000 | 0.0000 |

这个结果说明：

- 在干净 GT 域，手工 explicit SE3 很强；
- 但这个优势不能直接迁移到 Human3R 输出域；
- 因此正式训练和推理更应该以 Human3R output-domain 为主；
- GT 域更适合作为 sanity check 或辅助监督，而不是直接替代 Human3R output-domain。

### 9.4 Viewer payload

导出目录：

```text
output/v10_bedlam21_minimal_validation/viewer_payload_fresh
```

已导出 7 个版本：

```text
target_original_human3r
raw_perturbed
fixed_explicit_se3
history_current_integrator
history_direct_residual_integrator
explicit_se3_residual_integrator
oracle_se3_upper
```

每个版本均包含 29 帧完整 payload：

```text
camera=29
smpl=29
depth=29
conf=29
color=29
```

dry-run 在 CPU 上加载多人 SMPL mesh 较慢，本轮没有等待完成；payload 文件完整，后续可以直接开 viewer 端口主观看。

## 10. 当前最小验证结论

这个最小验证支持当前 V10 主线：

```text
Human3R output-domain
  + oracle boundary
  + learned history-current coarse SE3
  + learned residual SE3
  + segment-level transform cache
```

现在可以确认：

1. 这个方法在严格流式 synthetic local-reset 设置下是可训练的。
2. 历史 global state 是核心信息，current-only 不够。
3. residual head 有增益，但当前增益还不算特别大，后续可以继续优化 feature 和 loss。
4. 直接手工 explicit SE3 在 Human3R 输出域不可靠。
5. 点云背景 cue 还没有进入训练；它是下一步增强项，不是当前最小验证成立的必要条件。

下一步建议：

```text
V10.1 = history-current direct + residual + compact scene cue
```

重点测试 scene cue 是否能改善：

- camera/scene 倾斜；
- 地面方向不一致；
- 人体对齐了但背景不连续；
- residual 增益不够大的问题。
