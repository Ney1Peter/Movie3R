# V8.6 Human Correction Ablation Plan

## 背景

当前 pose-only correction 已经证明：

```text
camera pose 有机会被修正到比较接近 GT。
```

但新的问题是：

```text
camera 对了，不代表人也对了。
```

Human3R 的输出里，camera、pointmap、SMPL 是几套相互关联但又独立的量：

```text
world_point = camera_pose @ pts3d_in_self_view
world_smpl  = camera_pose @ SMPL(smpl_pose, smpl_shape, smpl_transl)
```

所以只改 `camera_pose` 时，点云和人都会被换到新的 world 坐标里，但它们各自在 camera space 内的深度和平移不会自动变对。

通俗地说：

```text
相机的位置修正了，
但“人在相机前方多远、偏左偏右多少”这个人体自己的位置参数还没修。
```

因此下一步要测试的是：

```text
能不能在 pose correction token 的基础上，把人体 SMPL 也一起修正。
```

当前阶段先不启用 ASIT/native SMPL，主线仍然只用：

```text
AvatarReX + THuman
SMPL-X-like smplx_* fields
pose-only correction baseline
```

## 测试目标

先拿一个前面已经分析过的大角度 AvatarReX clip 做单样本 overfit。

这个实验不是为了证明泛化，而是为了回答三个问题：

```text
1. 当前 token 里有没有足够信息把人修回来？
2. 只是训练不够，还是结构本身缺了 human correction 分支？
3. 应该先修 SMPL transl、root rotation，还是 human latent / human head？
```

所有实验都必须用同一个 viewer 协议：

```text
GT camera: red
raw Human3R camera: gray
corrected camera: yellow
raw / corrected / GT 必须同一个 gauge
推理过程不能用 GT
GT 只用于 loss、metric、viewer 参考
```

## 总体结构设想

当前主线是：

```text
image / camera / human / pose tokens
  + pose correction token
  + recurrent state
  -> decoder
  -> corrected pose token
  -> pose residual head
  -> corrected camera pose
```

下一步希望扩展成：

```text
image / camera / human / pose tokens
  + pose-human correction token
  + recurrent state
  -> decoder
  -> corrected pose token + corrected human relation token
  -> pose residual head
  -> corrected camera pose

  -> human residual head
  -> corrected SMPL transl / root / human latent
  -> corrected SMPL
```

通俗地说：

```text
原来只让新增分支告诉模型“相机该怎么动”。
现在要让它同时告诉模型“人自己在相机坐标里也该怎么动”。
```

## Token 设计

暂时命名为：

```text
A_hcorr_t
```

含义是当前帧的 human-aware correction token。

它可以由下面几类信息组成：

| 组成 | 通俗解释 | 推理时是否可用 | 作用 |
|---|---|---:|---|
| current human token | 当前图里模型看到的人 | 是 | 告诉模型当前人长什么样、人体大概在哪里 |
| current body-part anchors | 躯干、骨盆、左右脚等人体部位锚点 | 是 | 给模型更稳定的人体参照点 |
| previous corrected human anchors | 上一帧修正后的身体锚点 | 是 | 告诉模型上一帧人在哪里 |
| previous corrected pose memory | 上一帧修正后的相机/pose 状态 | 是 | 告诉模型上一帧相机怎么看这个场景 |
| current-vs-history relation | 当前人和历史人之间的关系 | 是 | 判断当前是不是漂了、漂了多少 |
| correction gate | 是否需要修正的开关 | 是 | 正常帧少修，跳变帧多修 |

这里要注意：

```text
不能把当前帧 GT SMPL、GT camera、shot label、GT ray_map 放进 token。
```

因为这些东西真实推理时拿不到。

## 实验设计

### Experiment A. Pose-only baseline

**做什么**

复现当前 pose-only 版本，不改人体输出，只修 camera pose。

**通俗解释**

这是对照组：

```text
只修相机，不修人。
```

如果 viewer 里相机变对了但人还偏，就说明后面 human correction 是必要的。

**看什么**

```text
camera trans / rot error
SMPL head world error
SMPL pelvis world error
SMPL mean joint world error
viewer 中人和场景是否对齐
```

**预期**

camera 指标可能变好，但 SMPL world 指标不一定好。

### Experiment B. 显式 SMPL translation residual

**做什么**

在当前 pose-only 基础上，加一个最简单的 human translation residual head：

```text
delta_smpl_transl = human_transl_residual_head(relation token)
corrected_smpl_transl = raw_smpl_transl + delta_smpl_transl
```

只改 `smpl_transl`，不改 body pose，不改 shape，不改 pointmap。

**通俗解释**

就是让模型学会：

```text
相机修完以后，人自己还应该往前/后/左/右移动多少。
```

这对应之前手动 oracle 实验里“给人加一个平移以后就更对”的现象。

**为什么先做它**

因为它最容易验证：

- 自由度只有 3 个数。
- 和当前错误现象直接对应。
- 不容易把人体姿态训坏。
- 如果单 clip 都 overfit 不动，说明 token 或 loss 有更基础的问题。

**Loss**

```text
L_smpl_head_world
L_smpl_pelvis_world
L_smpl_mean_joint_world
L_smpl_transl_residual_small
L_camera_pose_keep
```

其中最关键是：

```text
corrected_camera_pose @ corrected_SMPL
  要接近
GT_camera_pose @ GT_SMPL
```

**成功标准**

```text
camera 仍然接近 GT
SMPL head / pelvis / mean joints 明显下降
viewer 中人回到正确位置
```

### Experiment C. SMPL translation + root rotation residual

**做什么**

在 B 的基础上，再预测人体 root orientation 的小残差：

```text
corrected_root_orient = raw_root_orient + delta_root_orient
```

**通俗解释**

B 只会把人整体搬过去，但不会转人。

C 是让模型额外学会：

```text
人整体应该朝哪个方向。
```

**为什么不是直接改全身 pose**

因为全身 body pose 自由度太高，很容易把手脚动作训坏。先只修 root，是比较稳的中间方案。

**Loss**

```text
L_smpl_head_world
L_smpl_pelvis_world
L_smpl_mean_joint_world
L_root_rot
L_residual_small
```

**成功标准**

如果 B 已经能修位置，但身体朝向或整体方向仍有问题，C 应该进一步改善。

### Experiment D. UniCon-style human latent residual

**做什么**

不直接改显式 `smpl_transl`，而是更接近 UniCon3R：

```text
decoder output human relation token
  -> human latent residual head
  -> delta_human_latent

refined_human_latent + delta_human_latent
  -> human head
  -> corrected SMPL
```

**通俗解释**

B/C 像是在 human head 输出后做一个明确的补丁。

D 更像 UniCon3R：

```text
先在隐式特征里把“人应该怎么修”表达出来，
再交给 human head 输出最终人体。
```

**为什么要做**

因为最终我们希望是 token/decoder 内部学会关系，而不是一个纯后处理。

**风险**

比 B/C 难 debug：

- latent residual 不一定对应明确的平移。
- 如果 human head 不配合，可能学不动。
- viewer 错了时更难判断是哪一层错。

**成功标准**

单 clip overfit 能达到接近 B 的效果，并且 correction 不是靠 GT 或后处理完成。

### Experiment E. Human head adapter / LoRA

**做什么**

在 D 或 B 的基础上，只微调 human head 的小模块：

```text
freeze backbone
freeze decoder
freeze main human head weights
train correction token
train residual head
train human head LoRA / adapter
```

**通俗解释**

human head 原来是 Human3R 学好的“人体输出器”。

现在新增 correction token 可能给了它新的信息，但原 human head 不一定知道怎么用。

所以加一个很小的 adapter，让它学会：

```text
看到 correction 信息时，怎么稍微调整人体输出。
```

**为什么不用全量微调**

全量微调 human head 风险大：

```text
可能修好了这个 clip，
但破坏原本 Human3R 的人体能力。
```

**成功标准**

比 D 更容易收敛，但 AAAA 或 already-good AABB 上不能乱改人。

### Experiment F. Pose head adapter / LoRA

**做什么**

在 human correction 基本可行后，再试 pose head 的小范围 adapter / LoRA。

**通俗解释**

pose head 负责相机位姿输出。

如果 human correction 和 camera correction 之间需要更强耦合，可以让 pose head 小范围适应 correction token。

**为什么优先级低**

因为现在主要问题已经不是相机完全修不动，而是：

```text
相机修了，人没跟上。
```

所以先动 human 分支更直接。

**成功标准**

相机指标不能变差，SMPL 指标要进一步变好。

### Experiment G. Pointmap / scene residual diagnostic

**做什么**

如果 B/C/D 能把人修对，但 pointcloud 仍然和人不一致，再考虑 pointmap 或 scene residual。

**通俗解释**

如果最后发现：

```text
相机对了，人也对了，但点云还飘。
```

那说明点云自己的 self-view depth / pointmap 也需要修。

**当前阶段先不做**

因为它自由度更高，容易把问题搞复杂。先把 camera + SMPL 讲清楚。

## Loss 设计说明

| Loss | 通俗解释 | 作用 |
|---|---|---|
| `L_camera_pose` | 相机别修坏 | 保持 pose correction 能力 |
| `L_smpl_head_world` | 头的位置要对 | 很直观地看人有没有偏 |
| `L_smpl_pelvis_world` | 身体中心要对 | 防止只对齐头但身体整体偏 |
| `L_smpl_mean_joint_world` | 全身平均要对 | 让整个人都对，而不是某个点对 |
| `L_smpl_transl_direct` | 直接监督人体平移补多少 | 帮 B 快速收敛 |
| `L_root_rot` | 人整体朝向要对 | 帮 C 修正人体方向 |
| `L_human_history_alignment` | 当前人和上一帧人要连续 | 防止跳变后人突然飞走 |
| `L_aaaa_noop` | 正常连续帧不要乱修 | 防止模型过度纠正 |
| `L_gate` | 学会什么时候该修 | drift 高 gate，正常低 gate |
| `L_residual_small` | 能不改就少改 | 防止模型到处乱动 |

最关键的训练目标不是只让 `smpl_transl` 数值像 GT，而是让最终 world-space 人体正确：

```text
corrected_world_smpl =
  corrected_camera_pose @ corrected_smpl_camera_space
```

它应该接近：

```text
gt_world_smpl =
  gt_camera_pose @ gt_smpl_camera_space
```

## 单 clip 实验顺序

建议按下面顺序做，每一步都保存 metric 和 viewer：

| 顺序 | 实验 | 目的 | 继续条件 |
|---:|---|---|---|
| 0 | 统一 baseline viewer | 确认 raw / corrected / GT 坐标完全一致 | viewer 正确 |
| 1 | A: pose-only baseline | 确认问题仍然是“相机对，人不对” | SMPL error 仍高 |
| 2 | B: `delta_smpl_transl` | 验证能否靠人体平移修回来 | SMPL world error 明显下降 |
| 3 | C: `delta_smpl_transl + delta_root_orient` | 看是否还需要整体朝向修正 | 比 B 更好或至少不差 |
| 4 | D: human latent residual | 验证 UniCon-style 隐式方案 | 接近 B/C 效果 |
| 5 | E: human head LoRA / adapter | 看 human head 小微调是否有帮助 | 不破坏 normal case |
| 6 | F: pose head LoRA / adapter | 只作为后置对照 | camera 不变差 |
| 7 | G: pointmap diagnostic | 判断是否还要修点云 | 仅当点云仍错 |

## 每个实验都要保存的指标

```text
camera_trans_error_raw
camera_trans_error_corrected
camera_rot_error_raw
camera_rot_error_corrected

smpl_head_world_error_raw
smpl_head_world_error_corrected
smpl_pelvis_world_error_raw
smpl_pelvis_world_error_corrected
smpl_mean_joint_world_error_raw
smpl_mean_joint_world_error_corrected

delta_smpl_transl_norm
delta_root_orient_norm
gate_mean
residual_norm
```

还要保存 viewer：

```text
GT camera red
Human3R raw camera gray
corrected camera yellow
raw SMPL
corrected SMPL
pointcloud
```

## 预期判断

### 如果 B 成功

说明问题很明确：

```text
当前结构缺的是 human translation residual。
```

后续就可以把 `delta_smpl_transl` 正式纳入 V8.6/V8.7 主线。

### 如果 B 不成功

说明可能是：

```text
token 里没有足够的人体/历史关系信息，
或者 loss / 坐标还有问题。
```

这时不要急着上 LoRA，应该先检查：

- `smpl_transl` 坐标系是不是 camera space。
- corrected camera 和 corrected SMPL 是否在同一个 gauge。
- loss 里的 GT SMPL world 是否算对。
- body-part anchor / history memory 是否真的进了 decoder。

### 如果 B 成功但 D 不成功

说明：

```text
显式 residual 能解决问题，
但 UniCon-style latent residual 还没学会。
```

这时可以保留 B 作为诊断上限，再改 D 的 token / head / loss。

### 如果 E 明显好于 D

说明：

```text
原 human head 不太会使用新的 correction 信息，
需要小范围 adapter / LoRA 帮它适配。
```

但仍然不建议全量微调 human head。

### 如果 F 才有效

说明：

```text
camera pose 和 human correction 的耦合更强，
pose head 也需要轻量适配。
```

但要小心：pose head 微调容易破坏原 Human3R 的稳定能力。

## 最推荐先做的版本

第一轮只做：

```text
A: pose-only baseline
B: pose-only + delta_smpl_transl
```

原因：

- 最快。
- 最容易解释。
- 和手动平移实验直接对应。
- 能明确判断“是不是缺 human translation residual”。

如果 B 在单 clip 上明显成功，再继续 C/D/E。

